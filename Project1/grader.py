import os, time
import numpy as np
from skimage import io

from prj1_S20131598 import edge

path_images = 'images'
path_GT = 'gt'

fnames_image = next(os.walk( path_images ))[2]
fnames_GT = next(os.walk( path_images ))[2]

fmeasure_sum = 0
t_elapsed = 0
for fname_image, fname_GT in zip(fnames_image,fnames_GT):
    image = io.imread( os.path.join(path_images,fname_image) )
    GT = io.imread( os.path.join(path_GT,fname_GT),as_grey=True )>0

    # your function "edge()" is supposed to return binary image
    # with 1 for edge and 0 for non-edge
    # height x width (2-dim, not 3-dim of h x w x 1)
    t_start = time.time()
    result = edge( image )
    t_elapsed += time.time()-t_start
    
    TP = np.bitwise_and( GT, result )
    nTP = TP.sum()
    nReal = GT.sum()
    nResponse = result.sum()
    precision = nTP / nResponse
    recall = nTP / nReal
    fmeasure = 2*precision*recall/(precision+recall)
    fmeasure_sum += fmeasure
    
fmeasure_avg = fmeasure_sum / len(fnames_image)

print( fmeasure_avg )
print( t_elapsed )

