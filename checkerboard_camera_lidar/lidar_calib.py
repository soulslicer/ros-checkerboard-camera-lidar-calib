#!/usr/bin/python
import cv2
import torch
import numpy as np
import pickle
import os
import rospy
import rospkg

path = rospkg.RosPack().get_path('checkerboard_camera_lidar') + "/"
import json
with open(path + 'datum.json') as json_file:
    datum = json.load(json_file)

calibpath = datum['save_location']
device = torch.device('cpu')
draw = [2,4,6]
guess = [0,0,0,1.5690,0,1.5949]
lr = 1.0e-5
#guess = datum['lidar_to_left_cam']

def process(datum):
    lidarLines = []
    for lidarLine in datum["lidarLines"]:
        lidarLines.append(torch.tensor(lidarLine[1].astype(np.float32)).to(device))
    datum["lidarLines"] = lidarLines
    datum["boardLines"] = torch.tensor(datum["boardLines"]).to(torch.float32).to(device)

def load_data(calibpath):
    left_datapoints = []
    right_datapoints = []
    leftToRight = None
    for (dirpath, dirnames, filenames) in os.walk(calibpath):
        for file in filenames:
            if "pickle" not in file:
                continue
            
            with open(calibpath + file, 'rb') as handle:
                datum = pickle.load(handle)

                process(datum["left"])
                left_datapoints.append(datum["left"])
                if 'right' in datum.keys():
                    process(datum["right"])
                    leftToRight = torch.tensor(datum["leftToRight"]).to(torch.float32).to(device)
                    right_datapoints.append(datum["right"])
    return left_datapoints, right_datapoints, leftToRight

def euler2mat(angle):
    """Convert euler angles to rotation matrix.

     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:,0], angle[:,1], angle[:,2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = torch.matmul(torch.matmul(xmat, ymat), zmat)
    return rotMat

def pose_vec2mat(vec):
    """
    Convert 6DoF parameters to transformation matrix.

    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:,3:]
    rot_mat = euler2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    ones = torch.tensor([0., 0., 0., 1.]).unsqueeze(0).unsqueeze(0).repeat(transform_mat.shape[0], 1, 1).to(vec.device)
    transform_mat_f = torch.cat([transform_mat, ones], dim=1)  # [B, 4, 4]
    return transform_mat_f

def proj_points(Pts, T, K):
    ProjM = torch.mm(K, T)
    projPoints = torch.mm(ProjM, Pts.T).T
    U = projPoints[:,0] / projPoints[:,2]
    V = projPoints[:,1] / projPoints[:,2]
    projPointsNorm = torch.cat([U.unsqueeze(1),V.unsqueeze(1)], dim=1)
    return projPointsNorm

def pt_to_line(x1,y1,x2,y2,x0,y0):
    return torch.abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) / torch.sqrt((y2-y1)**2 + (x2-x1)**2)

import torch.nn as nn
class DefaultModel(nn.Module):
    def __init__(self, guess, K):
        super(DefaultModel, self).__init__()

        self.T = torch.tensor(guess).unsqueeze(0).to(device)
        self.T.requires_grad = True

        self.K = K

    def forward(self, datapoints, add_t, draw, text):
        Mat_temp = pose_vec2mat(self.T).squeeze(0)
        Mat = torch.mm(add_t, Mat_temp)

        #N = float(len(datapoints))
        N = 0.
        loss = 0.
        i=-1
        for datum in datapoints:
            i+=1

            boardLines = datum["boardLines"]
            lidarLines = datum["lidarLines"]
            cvImgDraw = None
            if draw[0] == -1:
                grayImg = datum["img"]
                cvImgDraw = cv2.cvtColor(grayImg, cv2.COLOR_GRAY2BGR)
            if i in draw:
                grayImg = datum["img"]
                cvImgDraw = cv2.cvtColor(grayImg, cv2.COLOR_GRAY2BGR)

            # Project Test
            errors = 0.
            boardColors = [(255,0,255), (0,255,255), (255,255,0), (0,200,0)]
            for lidarLine, boardLine, boardColor in zip(lidarLines, boardLines, boardColors):
                proj_pixels = proj_points(lidarLine, Mat, self.K)
                distances = pt_to_line(boardLine[0,0], boardLine[0,1],boardLine[1,0], boardLine[1,1],proj_pixels[:,0],proj_pixels[:,1])
                loss += torch.sum(distances)
                N += distances.shape[0]

                # Draw
                if cvImgDraw is not None:
                    cv2.line(cvImgDraw, (boardLine[0,0], boardLine[0,1]), (boardLine[1,0], boardLine[1,1]), boardColor, 2)
                    for pixel in proj_pixels.tolist():
                        cv2.circle(cvImgDraw,(int(pixel[0]), int(pixel[1])), 3, boardColor, -1)

            # Draw
            if cvImgDraw is not None:
                if draw[0] == -1:
                    cv2.imshow(text + "win", cvImgDraw)
                    cv2.waitKey(0)
                else:
                    cv2.imshow(text + str(i), cvImgDraw)
                    cv2.waitKey(1)

        # Divide
        loss /= N

        return loss

# Load Data
left_datapoints, right_datapoints, leftToRight = load_data(calibpath)
if "K" in left_datapoints[0]:
    print("Set K")
    K = left_datapoints[0]["K"]
    K = np.hstack((K, np.expand_dims(np.zeros(3), axis=1)))
    K = torch.tensor(K).to(torch.float32).to(device)
    rotateT = torch.tensor(left_datapoints[0]["rotateT"].astype(np.float32)).to(device)

# Optimize
model = DefaultModel(guess, K).to(device)
weights = torch.tensor([1, 1, 1, 0.5, 0.5, 0.5]).to(device)
gamma = weights * lr
prev_loss = 100.
i=-1
while 1:
    i+=1

    if len(right_datapoints) == 0:
        left_T = torch.eye(4)
        left_loss = model(left_datapoints, left_T, draw=draw, text="left")
        loss = left_loss
        right_loss = 0
    else:
        left_T = torch.eye(4)
        right_T = leftToRight
        left_loss = model(left_datapoints, left_T, draw=draw, text="left")
        right_loss = model(right_datapoints, right_T, draw=draw, text="right")
        loss = left_loss*0.5 + right_loss*0.5

    loss.backward()

    # Metrics
    diff = torch.abs(prev_loss-loss.detach())
    if(loss > prev_loss):
        lr = lr * 0.5
        gamma = weights * lr
        print("HALF: " + str(lr))
    prev_loss = loss.detach()
    print((left_loss, right_loss), diff)

    # Update Gradient
    with torch.no_grad():
        model.T = model.T - gamma * model.T.grad
    model.T.requires_grad = True

    # Print result
    if i % 200 == 0:
        print(model.T)
        lidarToCamera = pose_vec2mat(model.T).squeeze(0).detach()
        lidarToCameraS = torch.mm(lidarToCamera, rotateT)
        lidarToCameraS = torch.inverse(lidarToCameraS)
        print(lidarToCameraS)

        # Break
        if diff < 1e-6:
            print("Lidar to Camera Transform:")
            print(lidarToCameraS)
            break

        



