import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

def SIFT_with_BF(queryimg, trainimg):
    start = time.time()

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(queryimg, None)
    kp2, des2 = sift.detectAndCompute(trainimg, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()

    matches = bf.knnMatch(des1, des2, k=2)

    matchesMask = [[0, 0] for i in range(len(matches))]

    for i, (m,n) in enumerate(matches):
        if m.distance < 0.75 * n.distance:
            matchesMask[i] = [1,0]

    draw_params = dict(singlePointColor=(255, 0, 0),
                       matchesMask = matchesMask,
                       flags=2)

    # cv2.drawMatchesKnn expects list of lists as matches
    img = cv2.drawMatchesKnn(queryimg, kp1, trainimg, kp2, matches, None, **draw_params)
    print("BF time : ", time.time() - start)

    plt.imshow(img),plt.show()
    cv2.imwrite('sift_with_bf.png', img)

def SIFT_with_FLANN(queryimg, trainimg):
    start = time.time()

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(queryimg, None)
    kp2, des2 = sift.detectAndCompute(trainimg, None)

    #FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, tress = 5)
    search_params = dict(checks=50) # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i, (m,n) in enumerate(matches):
        if m.distance < 0.75 * n.distance:
            matchesMask[i] = [1,0]

    draw_params = dict(singlePointColor = (255, 0, 0),
                   matchesMask = matchesMask,
                   flags = 2)

    img = cv2.drawMatchesKnn(queryimg, kp1, trainimg, kp2, matches, None, **draw_params)
    print("FLANN time : ", time.time() - start)

    plt.imshow(img),plt.show()
    cv2.imwrite('sift_with_flann.png', img)

# Debugging
if __name__  == "__main__":
    #img1 = cv2.imread('box.png', 0)  # queryimage
    #img2 = cv2.imread('box_in_scene.png', 0)  # trainimagepng

    img1 = cv2.imread('monalisa.png', 0)  # queryimage
    img2 = cv2.imread('monalisa_scene.jpg', 0)  # trainimage

    SIFT_with_BF(img1, img2)
    SIFT_with_FLANN(img1, img2)

