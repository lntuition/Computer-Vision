import numpy as np
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter, convolve
from scipy.ndimage.morphology import grey_dilation
from skimage import io
from skimage.draw import circle

def harris(image, sigma=2, radius=3, alpha=0.04, thresh=0.2):
    kernel_deriv = np.array([[1,0,-1],
                             [2,0,-2],
                             [1,0,-1]])
    
    img_deriv_x = convolve(image, kernel_deriv, mode='constant', cval=0)
    img_deriv_y = convolve(image, kernel_deriv.transpose(), mode='constant', cval=0)

    igx = gaussian_filter(img_deriv_x ** 2, sigma, mode='constant', cval=0)
    igy = gaussian_filter(img_deriv_y ** 2, sigma, mode='constant', cval=0)
    igxy = gaussian_filter(img_deriv_x * img_deriv_y, sigma, mode='constant', cval=0)

    response = (igx * igy - igxy ** 2) - alpha * (igx + igy) ** 2
    response_dilated = grey_dilation(response, size=radius)

    img_max = response == response_dilated
    img_threshed = response > thresh

    img_corner = np.logical_and(img_max, img_threshed)

    # Modify intermediate images to save image file
    img_deriv_x /= max(abs(img_deriv_x.max()), abs(img_deriv_x.min()))
    img_deriv_y /= max(abs(img_deriv_y.max()), abs(img_deriv_y.min()))

    igx /= max(abs(igx.max()), abs(igx.min()))
    igy /= max(abs(igy.max()), abs(igy.min()))
    igxy /= max(abs(igxy.max()), abs(igxy.min()))

    img_threshed = (img_threshed * 255).astype('uint8')
    img_corner = (img_corner * 255).astype('uint8')

    return img_deriv_x, img_deriv_y, igx, igy, igxy, img_threshed, img_corner

if __name__ == "__main__":

    prefix = "geometry"
    postfix = ".jpg"
    img_input = io.imread(prefix + postfix, as_grey=True)
    img_ix, img_iy, img_igx, img_igy, img_igxy, img_threshed, img_corner = harris(img_input)

    idx = np.nonzero(img_corner)
    for (r, c) in zip(idx[0], idx[1]):
        rr, cc = circle(r, c, 5)
        try:
            img_input[rr, cc] = 0
        except:
            pass

    # Save intermediate images
    io.imsave(prefix + "_ix" + postfix, img_ix)
    io.imsave(prefix + "_iy" + postfix, img_iy)

    io.imsave(prefix + "_ix2" + postfix, img_ix ** 2)
    io.imsave(prefix + "_iy2" + postfix, img_iy ** 2)

    io.imsave(prefix + "_igx" + postfix, img_igx)
    io.imsave(prefix + "_igy" + postfix, img_igy)
    io.imsave(prefix + "_igxy" + postfix, img_igxy)

    io.imsave(prefix + "_cornerness_function_thresh" + postfix, img_threshed)
    io.imsave(prefix + "_non_maxima_suppression" + postfix, img_corner)

    # Save Result image
    io.imsave(prefix + "_result" + postfix, img_input)
