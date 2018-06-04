import numpy as np
from skimage import io,color
from skimage import filters as skifilters
from scipy.ndimage import filters

def edge(img):
    img = color.rgb2gray(img)

    # Suppress noise
    img = filters.gaussian_filter(img, sigma=2)
    height, width = img.shape
    shape = (height, width)

    img = np.lib.pad(img, 1, 'edge')
    gx = np.zeros(shape, dtype='float64')
    gy = np.zeros(shape, dtype='float64')

    # Calculate gradient and direction
    for y in range(1, height+1):
        for x in range(1, width+1):
            gx[y - 1, x - 1] = (img[y - 1, x - 1] + 2 * img[y, x - 1] + img[y + 1, x - 1]) - \
                               (img[y - 1, x + 1] + 2 * img[y, x + 1] + img[y + 1, x + 1])
            gy[y - 1, x - 1] = (img[y + 1, x - 1] + 2 * img[y + 1, x] + img[y + 1, x + 1]) - \
                               (img[y - 1, x - 1] + 2 * img[y - 1, x] + img[y - 1, x + 1])

    gmag = np.sqrt(gx**2 +gy**2)
    theta = np.arctan2(gy, gx)
    theta = np.rad2deg(theta) % 180

    # Apply non-maximum suppression
    gmax = np.zeros(shape, dtype='float64')
    for y in range(height):
        for x in range(width):
            try:
                if 22.5 <= theta[y,x] < 67.5:
                    if gmag[y,x] > gmag[y-1,x+1] and gmag[y,x] > gmag[y+1,x-1]:
                        gmax[y,x] = gmag[y,x]
                elif 67.5 <= theta[y,x] < 112.5:
                    if gmag[y,x] > gmag[y-1,x] and gmag[y,x] > gmag[y+1,x]:
                        gmax[y,x] = gmag[y,x]
                elif 112.5 <= theta[y,x] < 157.5:
                    if gmag[y,x] > gmag[y-1,x-1] and gmag[y,x] > gmag[y+1,x+1]:
                        gmax[y,x] = gmag[y,x]
                else:
                    if gmag[y,x] > gmag[y,x-1] and gmag[y,x] > gmag[y,x+1]:
                        gmax[y,x] = gmag[y,x]
            except IndexError as e:
                pass

    # Thresholding
    high = 0.130
    low = 0.065

    line = np.zeros(shape, dtype='uint8')
    line[np.where(gmax > high)] = 255 # Strong
    line[np.where((gmax >= low) & (gmax <= high))] = 50 # Weak

    # Connecting edges
    dy = (0, -1, -1, -1, 0, 1, 1, 1)
    dx = (1, 1, 0, -1, -1, -1, 0, 1)
    for y in range(height):
        for x in range(width):
            if line[y, x] == 50:
                try:
                    for i in range(8):
                        if line[y + dy[i], x + dx[i]] == 255:
                            line[y, x] = 255
                            break
                    else:
                        line[y, x] = 0
                except IndexError as e:
                    pass
    return line > 0

# Debugging
if __name__  == "__main__":
    import os, time

    img = io.imread(os.path.join('images', '001.jpg'))
    t_start = time.time()
    res = edge(img)
    t_elapsed = time.time()-t_start
    GT = io.imread(os.path.join('gt', '001.jpg'), as_grey=True) > 0
    TP = np.bitwise_and( GT, res )
    nTP = TP.sum()
    nReal = GT.sum()
    nResponse = res.sum()
    precision = nTP / nResponse
    recall = nTP / nReal
    fmeasure = 2*precision*recall/(precision+recall)

    print(fmeasure)
    print(t_elapsed)