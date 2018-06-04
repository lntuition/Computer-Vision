import numpy as np
from skimage import io, feature, color
from scipy import ndimage
import math, operator

def hough(image, threshold=0.90):
    eps = 1e-6

    # edge detection
    image = color.rgb2gray(image)
    image = feature.canny(image)

    # prepare Hough space
    rhoLimit = math.floor(np.linalg.norm(image.shape))
    rhoCount = rhoLimit*2 + 1
    thetaSamplingFreq = 0.0025
    thetaCount = math.floor(math.pi / thetaSamplingFreq)
    houghSpace = np.zeros((rhoCount, thetaCount))

    # pre-calculated trigonometric functions
    arrayCos = np.zeros(thetaCount)
    arraySin = np.zeros(thetaCount)
    theta = 0
    for i in range(thetaCount):
        arrayCos[i] = math.cos(theta)
        arraySin[i] = math.sin(theta)
        theta += thetaSamplingFreq

    # Run Hough Transform
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y,x] != 0:
                for i in range(thetaCount):
                    rho = math.floor(x*arrayCos[i] + y*arraySin[i])
                    houghSpace[rho+rhoLimit,i] += 1

    # Scale to 0~1
    houghSpace /= houghSpace.max()

    # threshold
    houghBin = houghSpace > threshold
    idx = np.nonzero(houghBin)

    # convert back
    rhoPoint = []
    thetaPoint = []
    dictPoint = {}
    for i in range(len(idx[0])):
        dictPoint[(abs(idx[0][i] - rhoLimit), idx[1][i] * thetaSamplingFreq)] = houghSpace[idx[0][i], idx[1][i]]

    dictPoint = sorted(dictPoint.items(), key=operator.itemgetter(1), reverse=True)
    for i in range(len(dictPoint)):
        rho, theta = dictPoint[i][0]
        rhoPoint.append(rho)
        thetaPoint.append(theta)
        
    # remove similar line
    drawedPoint = []
    rhoDrawedThresh = 5
    thetaDrawedThresh = 5 * thetaSamplingFreq

    # draw lines
    maxValue = np.iinfo('uint16').max
    edgeImg = image * maxValue
    houghImg = np.zeros(image.shape, dtype='uint16')
    for i in range(len(thetaPoint)):

        isDrawed = False
        for rho, theta in drawedPoint:
            if (abs(rho - rhoPoint[i]) < rhoDrawedThresh or abs(rho - rhoLimit - rhoPoint[i]) < rhoDrawedThresh) \
                    and (abs(theta - thetaPoint[i]) < thetaDrawedThresh or abs(theta - thetaCount - thetaPoint[i]) < thetaDrawedThresh):
                isDrawed = True
                break
        drawedPoint.append((rhoPoint[i], thetaPoint[i]))
        if isDrawed: continue

        for x in range(image.shape[1]):
            y = math.floor((rhoPoint[i] - x*math.cos(thetaPoint[i])) / (math.sin(thetaPoint[i]) + eps))
            if y < 0 or y >= image.shape[0]:
                continue
            houghImg[y,x] = maxValue
            
        for y in range(image.shape[0]):
            x = math.floor((rhoPoint[i] -  y*math.sin(thetaPoint[i])) / (math.cos(thetaPoint[i]) + eps))
            if x < 0 or x >= image.shape[1]:
                continue
            houghImg[y,x] = maxValue

    return edgeImg, houghSpace, houghBin, houghImg

# For debugging
if __name__ == "__main__":
    img = io.imread("test.png")

    edgeImg, houghSpace, houghBin, houghImg = hough(img)

    #io.imsave("test_edge.png", edgeImg)
    io.imsave("test_space.png", houghSpace)
    #io.imsave("test_bin.png", houghBin)
    io.imsave("test_hough.png", houghImg)
