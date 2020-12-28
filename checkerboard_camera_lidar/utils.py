# Params Lib Import
import rospkg
import sys
devel_folder = rospkg.RosPack().get_path('params_lib').split("/")
devel_folder = '/'.join(devel_folder[:len(devel_folder)-2]) + "/devel/lib/"
sys.path.append(devel_folder)
import params_lib_python
import numpy as np
from sensor_msgs.msg import PointCloud2
import rospy
import tf
import copy
from tf.transformations import quaternion_matrix
from cv_bridge import CvBridge
import image_geometry
import time
import os

class LidarCalibrator():

    def __init__(self, datum):
        self.datum = datum
        self.bridge = CvBridge()

        # self.store = args[0][0].store
        
        self.n_cols = self.datum['cols']
        self.n_rows = self.datum['rows']
        self.dim = self.datum['dim']
        self.db = []
        self.leftToRight = None

        # self.calib_cloud_pub = rospy.Publisher('/calib_cloud', PointCloud2, queue_size=1)
        # self.lidar_cloud_pub = rospy.Publisher('/lidar_cloud', PointCloud2, queue_size=1)
        # self.orig_cloud_pub = rospy.Publisher('/orig_cloud_pub', PointCloud2, queue_size=1)

        # Other Parameters


    def save_data(self):

        # Make or delete folder
        import os
        import shutil
        import pickle
        from random import randrange
        savepath = self.datum['save_location']
        try: shutil.rmtree(savepath)
        except: pass
        os.mkdir(savepath)

        # Write
        added_ints = []
        i=0
        for datum, cvImgDraw in self.db:
            # i = randrange(100000)
            # if i in added_ints:
            #     i = randrange(100000)
            # added_ints.append(i)
            cv2.imwrite(savepath + str(i) + ".png", cvImgDraw)
            with open(savepath + str(i) + '.pickle', 'wb') as handle:
                pickle.dump(datum, handle, protocol=pickle.HIGHEST_PROTOCOL)
            i+=1

    def extract_data(self, cammsg, caminfomsg, lidarPoints, lidarToCamera):
        # Modules
        cam_model = image_geometry.PinholeCameraModel()
        ext_calib = params_lib_python.ExtrinsicCalibration((self.n_cols, self.n_rows), self.dim)

        # Image Loading
        cvImg = self.bridge.imgmsg_to_cv2(cammsg, desired_encoding='passthrough')
        cam_model.fromCameraInfo(caminfomsg)
        cam_model.rectifyImage(cvImg, 0)
        cvImg = cv2.remap(cvImg, cam_model.mapx, cam_model.mapy, 1)
        grayImg = cv2.cvtColor(cvImg, cv2.COLOR_BGR2GRAY)

        # Extrinsics World Points
        cvImgDraw = cvImg.copy()
        #K = self.cam_model.K
        K = cam_model.P[0:3,0:3].copy()
        ext_calib.setCameraMatrix(K)
        T = np.eye(4)
        cameraPoints, corners = ext_calib.calibrateExtrinsics(cvImg, cvImgDraw, T, False)
        if cameraPoints.shape[0] == 0:
            time.sleep(1)
            cv2.putText(cvImgDraw,'Board not found',(10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            return cvImgDraw, corners, None

        # Get Board Pixel Lines
        projPoints, boardLines = board_lines(cameraPoints, T, K, cvImgDraw, squareSize=(self.n_cols,self.n_rows), squareDim=self.dim, offsets=self.datum["offsets"])

        # Mask the image
        rect = bbox(projPoints).astype(np.uint64)
        exp = 50
        maskImg = np.zeros((cvImgDraw.shape[0], cvImgDraw.shape[1])).astype(np.uint8)
        cv2.rectangle(maskImg, (int(rect[0,0]-exp),int(rect[1,0]-exp)), (int(rect[0,1]+exp),int(rect[1,1]+exp)), (1), -1) 

        # Detect Raw Image Lines
        monoImg = normalize_img(cvImg)
        rawLines = line_detect(monoImg, maskImg, None)

        # Associate Lines
        trueBoardLines = associate_lines(boardLines, rawLines, cvImgDraw)

        # Shift the Lidar Points to cull it
        jeepHeight = self.datum['lidar_height']
        lidarAngle = self.datum['lidar_tilt']
        rotateT = compute_transform(0., 0., 0., 0., lidarAngle, 0.)
        lidarPoints = np.matmul(rotateT, lidarPoints.T).T

        # Need an Initial guess for transform?
        # lidarToCamera = torch.tensor([-0.0, -0.1422, -0.2268,  1.5621,  0.0038,  1.6262]).unsqueeze(0)
        # lidarToCamera = pose_vec2mat(lidarToCamera).squeeze(0).numpy()
        cameraToLidar = np.linalg.inv(lidarToCamera)
        transCameraPoints = np.matmul(cameraToLidar, cameraPoints.T).T

        # Centroid for Lidar
        centroid = np.mean(transCameraPoints, axis=0)
        #print(centroid)
        if centroid[0] <= self.datum['minimum_zdist']:
            cv2.putText(cvImgDraw,'minimum_zdist too high',(10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            return cvImgDraw, corners, None
        
        # # Publish Trans Camera Points
        # cameraPointsRos = params_lib_python.convertXYZtoPointCloud2(transCameraPoints)
        # cameraPointsRos.header.frame_id = "lidar0/os_lidar"
        # self.calib_cloud_pub.publish(cameraPointsRos)

        # Cull Cloud
        lidarPointsIS = lidarPoints
        fitLimit = 1.5
        culledLidarPointsIS = lidarPointsIS
        culledLidarPointsIS = culledLidarPointsIS[(culledLidarPointsIS[:,0] > centroid[0] - fitLimit) & (culledLidarPointsIS[:,0] < centroid[0] + fitLimit)]
        culledLidarPointsIS = culledLidarPointsIS[(culledLidarPointsIS[:,1] > centroid[1] - fitLimit) & (culledLidarPointsIS[:,1] < centroid[1] + fitLimit)]
        culledLidarPointsIS = culledLidarPointsIS[(culledLidarPointsIS[:,2] > -jeepHeight)]
        culledLidarPoints = culledLidarPointsIS[:, 0:4].copy()
        culledLidarPointsIS = np.delete(culledLidarPointsIS, 3, 1) # Remove 1 index

        # # Publish Lidar Points
        # lidarPointsRos = params_lib_python.convertXYZtoPointCloud2(culledLidarPoints)
        # lidarPointsRos.header.frame_id = "lidar0/os_lidar"
        # self.orig_cloud_pub.publish(lidarPointsRos)

        # Fit Plane
        planeLidarPoints = params_lib_python.planeFit(culledLidarPoints, self.datum['plane_fit_ransac_error'])

        # Fit Contour
        convexLidarPoints = params_lib_python.concaveFit(planeLidarPoints)

        # Reproject Contour to Test
        # proj_points(planeLidarPoints, lidarToCamera, K, cvImgDraw)

        # Lines Fit
        lidarLines = params_lib_python.linesFit(convexLidarPoints, self.datum['line_fit_ransac_error'])
        if len(lidarLines) != 4 or len(trueBoardLines) != 4:
            cv2.putText(cvImgDraw,'Lidar lines not found',(10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            return cvImgDraw, corners, None

        # Lidar Line Association
        lidarLines = lidar_line_association(trueBoardLines, lidarLines, cvImgDraw, lidarToCamera, K)
        
        # Plot Lidar Lines
        for lidarLine, lineColor in zip(lidarLines, boardColors):
            proj_points(lidarLine[1], lidarToCamera, K, cvImgDraw, color=lineColor)

        # Fit Area
        area = poly_area(convexLidarPoints[:,0:3].tolist())
        #print("Area: " + str(area))
        if abs(area - self.datum['board_area']) > self.datum['minimum_area_diff']:
            cv2.putText(cvImgDraw,'Area Too Small',(10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            return cvImgDraw, corners, None

        # # Publish
        # convexLidarPointsRos = params_lib_python.convertXYZtoPointCloud2(convexLidarPoints)
        # convexLidarPointsRos.header.frame_id = "lidar0/os_lidar"
        # self.lidar_cloud_pub.publish(convexLidarPointsRos)

        # Datum
        datum = dict()
        datum["lidarLines"] = lidarLines
        datum["boardLines"] = trueBoardLines
        datum["img"] = grayImg
        datum["K"] = K
        datum["rotateT"] = rotateT
        cv2.putText(cvImgDraw,'Good',(10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        #datum["initGuess"] = lidarToCamera

        return cvImgDraw, corners, datum

    def get_transform(self, to_frame, from_frame):
        listener = tf.TransformListener()
        listener.waitForTransform(to_frame, from_frame, rospy.Time(), rospy.Duration(4.0))
        (trans,rot) = listener.lookupTransform(to_frame, from_frame, rospy.Time(0))
        matrix = quaternion_matrix(rot)
        matrix[0:3, 3] = trans
        return matrix.astype(np.float32)

    def handle_msg(self, msg):
        # Mono Case
        if not self.datum["stereo"]:
            left_cammsg = msg[0]
            left_caminfomsg = msg[1]
            lidarmsg = msg[2]        

            # Transform
            lidarToLeftCamera = torch.tensor(self.datum["lidar_to_left_cam"]).unsqueeze(0)
            lidarToLeftCamera = pose_vec2mat(lidarToLeftCamera).squeeze(0).numpy()

            # Cloud Loading
            lidarPoints = params_lib_python.convertPointCloud2toXYZ(lidarmsg)

            # Process
            left_cvImgDraw, left_corners, left_datum = self.extract_data(left_cammsg, left_caminfomsg, lidarPoints, lidarToLeftCamera)

            # Imshow with Keyboard stuff
            cv2.imshow("win", cv2.resize(left_cvImgDraw, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_LINEAR))
            key = cv2.waitKey(1)
            if key == 32 and left_datum is not None:
                print("Noted")

                # Store
                combined_datum = dict()
                combined_datum["left"] = left_datum

                self.db.append((combined_datum, left_cvImgDraw))

            # Save and quit
            if key == 119:
                print("Saved")
                self.save_data()
                os._exit(1)
            
        else:

            # Messages
            left_cammsg = msg[0]
            left_caminfomsg = msg[1]
            right_cammsg = msg[2]
            right_caminfomsg = msg[3]
            lidarmsg = msg[4]

            # Left To Right Transform
            if self.leftToRight is None:
                # Generate the Fake transform?
                cam_model = image_geometry.PinholeCameraModel()
                cam_model.fromCameraInfo(right_caminfomsg)
                fx = cam_model.P[0,0]
                baseline = cam_model.P[0,3]
                tx = baseline / fx
                self.leftToRight = np.eye(4).astype(np.float32)
                self.leftToRight[0,3] = -tx

            # Transform
            lidarToLeftCamera = torch.tensor(self.datum["lidar_to_left_cam"]).unsqueeze(0)
            lidarToLeftCamera = pose_vec2mat(lidarToLeftCamera).squeeze(0).numpy()
            lidarToRightCamera = np.matmul(self.leftToRight, lidarToLeftCamera)

            # Cloud Loading
            lidarPoints = params_lib_python.convertPointCloud2toXYZ(lidarmsg)

            # Process
            start = time.time()
            left_cvImgDraw, left_corners, left_datum = self.extract_data(left_cammsg, left_caminfomsg, lidarPoints, lidarToLeftCamera)
            right_cvImgDraw, right_corners, right_datum = self.extract_data(right_cammsg, right_caminfomsg, lidarPoints, lidarToRightCamera)
            end = time.time()

            # Stack Images
            cvImgDraw = np.hstack((left_cvImgDraw, right_cvImgDraw))

            # Imshow with Keyboard stuff
            cv2.imshow("win", cv2.resize(cvImgDraw, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_LINEAR))
            key = cv2.waitKey(1)
            if key == 32 and left_datum is not None:
                print("Noted")

                # Store
                combined_datum = dict()
                combined_datum["left"] = left_datum
                combined_datum["right"] = right_datum
                combined_datum["leftToRight"] = self.leftToRight

                self.db.append((combined_datum, cvImgDraw))

            # Save and quit
            if key == 119:
                print("Saved")
                self.save_data()
                os._exit(1)

import numpy as np
from sensor_msgs.msg import PointCloud2
import rospy
import tf
import copy
import cv2
import torch

boardColors = [(255,0,255), (0,255,255), (255,255,0), (0,200,0)]

def compute_transform(x, y, z, roll, pitch, yaw):
    transform = np.eye(4)
    transform = tf.transformations.euler_matrix(yaw*np.pi/180.,pitch*np.pi/180.,roll*np.pi/180.)
    transform[0,3] = x
    transform[1,3] = y
    transform[2,3] = z
    return np.array(transform)

def unit_normal(a, b, c):
    x = np.linalg.det([[1,a[1],a[2]],
         [1,b[1],b[2]],
         [1,c[1],c[2]]])
    y = np.linalg.det([[a[0],1,a[2]],
         [b[0],1,b[2]],
         [c[0],1,c[2]]])
    z = np.linalg.det([[a[0],a[1],1],
         [b[0],b[1],1],
         [c[0],c[1],1]])
    magnitude = (x**2 + y**2 + z**2)**.5
    return (x/magnitude, y/magnitude, z/magnitude)

def poly_area(poly):
    if len(poly) < 3: # not a plane - no area
        return 0
    total = [0, 0, 0]
    N = len(poly)
    for i in range(N):
        vi1 = poly[i]
        vi2 = poly[(i+1) % N]
        prod = np.cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]
    result = np.dot(total, unit_normal(poly[0], poly[1], poly[2]))
    return abs(result/2)

def normalize_img(img):
    mono = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mono = mono.astype(np.float32)/255.
    #mono *= (1.0/mono.max())
    mono = (mono - mono.min())/(mono.max() - mono.min())
    mono = mono*255
    mono = mono.astype(np.uint8)
    return mono

def line_detect(img, maskImg, drawImg):
    # https://stackoverflow.com/questions/45322630/how-to-detect-lines-in-opencv
    gray = img
    
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

    blur_gray = blur_gray * maskImg

    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    from random import randrange
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    points = []
    i=0
    for line in lines:
        i+=1
        j=0
        for x1, y1, x2, y2 in line:
            j+=1
            points.append(((x1 + 0.0, y1 + 0.0), (x2 + 0.0, y2 + 0.0)))
            color = (i*10%255, j*10%255, 255)
            if drawImg is not None:
                cv2.line(drawImg, (x1, y1), (x2, y2), color, 1)

    return points

def board_lines(cameraPoints, T, K, cvImgDraw, squareSize=(5,4), squareDim=0.2, offsets=[0.02, 0.02, 0.06, 0.03, 0.02, 0.02, 0.06, 0.03]):
    Tinv = np.linalg.inv(T)
    boardPoints = np.matmul(Tinv, cameraPoints.T).T
    tL = [boardPoints[0,0] + squareDim*squareSize[0] + offsets[0], boardPoints[0,1] - squareDim - offsets[1], 0, 1]
    tR = [boardPoints[0,0] - squareDim - offsets[2], boardPoints[0,1] - squareDim - offsets[3], 0, 1]
    bL = [boardPoints[0,0] + squareDim*squareSize[0] + offsets[4], boardPoints[0,1] + squareDim*squareSize[1] + offsets[5], 0, 1]
    bR = [boardPoints[0,0] - squareDim - offsets[6], boardPoints[0,1] + squareDim*squareSize[1] + offsets[7], 0, 1]
    cornerPoints = [tL, tR, bL, bR]
    cornerPoints = np.array(cornerPoints)
    Kx = np.hstack((K, np.expand_dims(np.zeros(3), axis=1)))
    I = np.eye(4)
    ProjM = np.matmul(Kx, T)
    projPoints = np.matmul(ProjM, cornerPoints.T).T
    projPoints[:,0] /= projPoints[:,2]
    projPoints[:,1] /= projPoints[:,2]
    projPoints = (projPoints[:,0:2]+0.0).astype(np.uint64).tolist()
    tL = tuple(projPoints[0])
    tR = tuple(projPoints[1])
    bL = tuple(projPoints[2])
    bR = tuple(projPoints[3])
    boardLines = [[bL, tL], [tL, tR], [tR, bR], [bR, bL]]
    if cvImgDraw is not None:
        try:
            # for line, color in zip(boardLines, boardColors):
            #     cv2.line(cvImgDraw, line[0], line[1], color, 2)
            
            for pixel, color in zip(projPoints, boardColors):
                cv2.circle(cvImgDraw,(int(pixel[0]), int(pixel[1])), 8, color, thickness=3)
        except:
            print("Pixel Error")
    projPoints = np.array(projPoints).astype(np.uint64)
    return projPoints, boardLines

def proj_points(Pts, T, K, cvImgDraw, color=(255,0,0)):
    Kx = np.hstack((K, np.expand_dims(np.zeros(3), axis=1)))
    ProjM = np.matmul(Kx, T)
    projPoints = np.matmul(ProjM, Pts.T).T
    projPoints[:,0] /= projPoints[:,2]
    projPoints[:,1] /= projPoints[:,2]
    projPoints = (projPoints[:,0:2]+0.0).astype(np.uint64)
    if cvImgDraw is not None:
        projPointsList = projPoints.tolist()
        for pixel in projPointsList:
            try:
                cv2.circle(cvImgDraw,(int(pixel[0]), int(pixel[1])), 3, color, -1)
            except:
                pass
    return projPoints

def bbox(points):
    a = np.zeros((2,2))
    a[:,0] = np.min(points, axis=0)
    a[:,1] = np.max(points, axis=0)
    return a

def pt_to_line(x1,y1,x2,y2,x0,y0):
    return np.abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) / np.sqrt((y2-y1)**2 + (x2-x1)**2)

def associate_lines(boardLines, rawLines, cvImgDraw):
    def compute_cost(a, b):
        # Board Line stable points
        ax1 = a[0,0]
        ay1 = a[0,1]
        ax2 = a[1,0]
        ay2 = a[1,1]

        # Raw Line
        bx1 = b[0,0]
        by1 = b[0,1]
        bx2 = b[1,0]
        by2 = b[1,1]

        cost = pt_to_line(bx1, by1, bx2, by2, ax1, ay1) + pt_to_line(bx1, by1, bx2, by2, ax2, ay2)

        return cost

    def line_intersection(line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            raise Exception('lines do not intersect')

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return (x, y)

    # Associate
    boardLines = np.array(boardLines)
    rawLines = np.array(rawLines)
    corrspLines = []
    i=-1
    for boardLine in boardLines:
        i+=1
        lowestCost = 1000000
        lowestLine = None
        for rawLine in rawLines:
            cost = compute_cost(boardLine, rawLine)
            if cost < lowestCost:
                lowestCost = cost
                lowestLine = rawLine

        if lowestLine is not None and lowestCost < 10:
            lowestLine = lowestLine.astype(np.uint64).tolist()
            #cv2.line(cvImgDraw, tuple(lowestLine[0]), tuple(lowestLine[1]), boardColors[i], 2)
            corrspLines.append(lowestLine)

    # If we do have 4 lines can we extend it out then by intersection!?
    if len(corrspLines) == 4:
        tL = line_intersection(corrspLines[0], corrspLines[1])
        tR = line_intersection(corrspLines[1], corrspLines[2])
        bR = line_intersection(corrspLines[2], corrspLines[3])
        bL = line_intersection(corrspLines[3], corrspLines[0])
        trueBoardLines = [[bL, tL], [tL, tR], [tR, bR], [bR, bL]]
        for line, color in zip(trueBoardLines, boardColors):
            cv2.line(cvImgDraw, line[0], line[1], color, 2)

        return trueBoardLines
    else:
        return []

def lidar_line_association(trueBoardLines, lidarLines, cvImgDraw, backToCam, K):
    allLineEdgePixels = []
    for lidarLine, color in zip(lidarLines, boardColors):
        lineEdgePoints = lidarLine[0]
        lineEdgePixels = proj_points(lineEdgePoints, backToCam, K, None, color=color)
        allLineEdgePixels.append(lineEdgePixels)
    sort_idx = []
    ignore_list = []
    for i in range(0, 4):
        boardLine = trueBoardLines[i]
        x1, y1, x2, y2 = boardLine[0][0], boardLine[0][1], boardLine[1][0], boardLine[1][1]
        lowestCost = 100000
        bestIdx = -1
        for j in range(0,4):
            if j in ignore_list:
                continue
            cost = 0
            for linePixel in allLineEdgePixels[j].tolist():
                x0 = linePixel[0]
                y0 = linePixel[1]
                cost += pt_to_line(x1,y1,x2,y2,x0,y0)
            if cost < lowestCost:
                lowestCost = cost
                bestIdx = j
        ignore_list.append(bestIdx)
        sort_idx.append(bestIdx)
    lidarLines = [lidarLines[sort_idx[0]], lidarLines[sort_idx[1]], lidarLines[sort_idx[2]], lidarLines[sort_idx[3]]]
    return lidarLines

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