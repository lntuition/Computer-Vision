import numpy as np
from skimage import io
from hw_S20131598 import hw1

fname_input = 'image1.jpg'
fname_output = 'image11.jpg'
fname_GT = 'imageGT.jpg'

hw1(fname_input, fname_output)

result = io.imread(fname_output)
GT = io.imread(fname_GT)
score = np.linalg.norm(result-GT)
print(score)
