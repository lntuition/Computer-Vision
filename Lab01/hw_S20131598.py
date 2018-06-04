from skimage import io

def hw1(fname_input, fname_output):
    image = io.imread(fname_input)
    image[:,:,0] = 0;
    io.imsave(fname_output, image)
