import numpy as np
import cv2
import random

def sift(img1, img2):
    # Find SIFT keypoints
    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Match SIFT keypoints
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Make matched keypoint pair list
    good = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            x1, y1 = kp1[m.queryIdx].pt            
            x2, y2 = kp2[m.trainIdx].pt
            good.append([x1, y1, x2, y2])
    
    return good

def ransac(pts, max_iters=100, num_points=4, threshold=5):
    best_num_inliers = 0
    best_h = None
    
    for i in range(max_iters):
        a = np.zeros((0, 9))
        num_inliers = 0

        # Sample points required to make homography
        for j in range(num_points):
            x1, y1, x2, y2 = pts[random.randrange(0, len(pts))]

            b = np.array([[x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2],
                          [0, 0, 0, x1, y1, 1, -y2*x1, -y2*x1, -y2]])
            a = np.concatenate((a, b))

        # Solve for model parameter
        u, s, v = np.linalg.svd(a)
        h = np.reshape(v[8], (3, 3))
        h /= h[2,2] # normalize

        # Score inliers of the model
        for x1, y1, x2, y2 in pts:
            dist = np.array([[x1], [y1], [1]])
            dist = np.dot(h, dist)
            dist /= dist[2]
            dist -= np.array([[x2], [y2], [1]])

            if np.linalg.norm(dist) < threshold:
                num_inliers += 1

        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_h = h

    return best_h

def stitch(img1, img2, homography):
    h1, w1 = img1.shape
    h2, w2 = img2.shape

    # Calculate Warp points
    top_left = np.dot(homography, np.array([0, 0, 1]))
    top_right = np.dot(homography, np.array([w2, 0, 1]))
    bottom_left = np.dot(homography, np.array([0, h2, 1]))
    bottom_right = np.dot(homography, np.array([w2, h2, 1]))

    # Normalize
    top_left /= top_left[2]
    top_right /= top_right[2]
    bottom_left /= bottom_left[2]
    bottom_right /= bottom_right[2]

    # Calculate Blended image size
    w_min = int(min(top_left[0], bottom_left[0], 0))
    w_max = int(max(top_right[0], bottom_right[0], w1))
    h_min = int(min(top_left[1], top_right[1], 0))
    h_max = int(max(bottom_left[1], bottom_right[1], h1))

    h, w = h_max - h_min, w_max - w_min
    img = np.zeros((h, w), np.int32)

    # Blending images
    homography= np.matrix([[1.0, 0.0, -w_min],
                           [0.0, 1.0, -h_min],
                           [0.0, 0.0, 1.0]]) * homography

    img = cv2.warpPerspective(img2, homography, (w, h))
    img[-h_min:-h_min+h1, -w_min:-w_min+w1] = img1

    return img

if __name__ == '__main__':
    # Need to change here(paramters)
    img1_name = 'images/railway/railway_2r.jpg'
    img2_name = 'images/railway/railway_3m.jpg'
    
    img1 = cv2.imread(img1_name, 0)
    img2 = cv2.imread(img2_name, 0)
             
    matched_pts = sift(img1, img2)
    homography = ransac(matched_pts)
    img_result = stitch(img2, img1, homography)

    # Debugging
    #cv2.imshow("debug", img_result)
    cv2.imwrite("result.jpg", img_result)
    
