import numpy as np
import cv2


def perspectiveWarp(frame):

    height, width = frame.shape[:2]
    y_sc = 0.6 
    x_sc = 0.3399 
    H2 = int(height * y_sc)
    W2_L = int(width * x_sc)
    W2_R = int(width * (1 - x_sc))
    
    src = np.float32([
        [W2_L, H2], 
        [W2_R, H2], 
        [width, height],
        [0, height]
    ])

    dst = np.float32([
        [0, 0],             
        [width, 0],         
        [width, height],    
        [0, height]         
    ])
    img_size = (width, height)
    matrix = cv2.getPerspectiveTransform(src, dst)
    birdseye = cv2.warpPerspective(frame, matrix, img_size)
    return birdseye


def adaptivePerspectiveWarp(frame, road_mask):
    height, width = frame.shape[:2]

    roi = road_mask[int(height*0.6):, :]

    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return frame

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    bottom_left = (x, int(height*0.6) + y + h)
    bottom_right = (x + w, int(height*0.6) + y + h)
    
    top_left = (x, int(height*0.6) + y)
    top_right = (x + w, int(height*0.6) + y)
    
    src = np.float32([top_left, top_right, bottom_right, bottom_left])
    dst = np.float32([[0,0],[width,0],[width,height],[0,height]])
    
    matrix = cv2.getPerspectiveTransform(src, dst)
    birdseye = cv2.warpPerspective(frame, matrix, (width, height))
    
    return birdseye


def stitch_robot_cameras(left_img, right_img, show_matches=False):
    gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(6000)
    kp_left, des_left = orb.detectAndCompute(gray_left, None)
    kp_right, des_right = orb.detectAndCompute(gray_right, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des_right, des_left, k=2)  # right → left

    good = [m for m, n in matches if m.distance < 0.8 * n.distance]

    if len(good) < 10:
        print("No point")
        return None

    pts_right = np.float32([kp_right[m.queryIdx].pt for m in good])
    pts_left = np.float32([kp_left[m.trainIdx].pt for m in good])

    H, mask = cv2.findHomography(pts_right, pts_left, cv2.RANSAC, 5.0)

    height, width = left_img.shape[:2]
    result = cv2.warpPerspective(right_img, H, (width + right_img.shape[1], height))
    result[0:height, 0:width] = left_img

    if show_matches:
        matched = cv2.drawMatches(right_img, kp_right, left_img, kp_left, good, None, flags=2)
        cv2.imshow("Feature Matches", matched)

    return result

