import numpy as np
from skimage import io
import timeit

IMG_NUM = 40
HEIGHT = 112
WIDTH = 92
SIZE = 10304 # HEIGHT * WIDTH
DIM = 10

def projection_mat_with_trick(eigfaces):
    # With original PCA, Compute the covariance matrix summation(XX^T)
    # But it is too large, Compute X^TX instead, then we easily get eigenvectors and eigenvalue

    cov = np.dot(eigfaces.transpose(), eigfaces)
    eigvals, eigvecs = np.linalg.eig(cov)

    idx = eigvals.argsort()[::-1]
    eigspace = np.dot(eigfaces, eigvecs)[:,idx][:,0:DIM].real

    return normalize_and_transpose_eigspace(eigspace, np.sqrt(eigvals[idx][0:DIM]))

def projection_mat_with_svd(eigfaces):
    # With SVD(Singular Value Decompostion), We do not need to calculate eigenvectors and eigenvalue
    
    U, S, V = np.linalg.svd(eigfaces, full_matrices=False)

    idx = S.argsort()[::-1]
    eigspace = np.dot(eigfaces, V.transpose())[:,idx][:,0:DIM].real
    
    return normalize_and_transpose_eigspace(eigspace, S[idx][0:DIM])

def normalize_and_transpose_eigspace(eigspace, size):
    # If we use trick or svd method, then calculated eigenvector is not unit vector. so we need to normalize.
    # Returning transform of projection matrix, we can reduce computation

    for i in range(SIZE):
        eigspace[i,:] /= size
    return eigspace.transpose()

def identify_imgs(original_imgs, mean_img, test_imgs, proj_mat, projected_mat, path):
    cnt = 0
    for i in range(IMG_NUM):
        diff = projected_mat - np.expand_dims(np.dot(proj_mat, (test_imgs[i] - mean_img)), 1)
        dist = np.linalg.norm(diff, axis=0)
        idx = dist.argsort()

        if idx[0] == i:
            cnt += 1

        io.imsave(path + '/s' + str(i+1) + '.3.tif', np.concatenate((original_imgs[idx[0]], original_imgs[idx[1]], original_imgs[idx[2]]), axis=1))
    return cnt / IMG_NUM * 100.0

if __name__ == "__main__":
    original_imgs = []
    train_imgs = []
    test_imgs = []
    centered_imgs = []
    mean_img = np.zeros([SIZE, ])
    t = ()

    '''
        Original PCA Algorithms
        1. Compute the mean vector of training imgs
        2. Center data(subtract the mean from all samples)
        3. Compute the covariance matrix summation
        4. Compute the eigenvectors and eigenvalue of 3.
        5. Select eigenvectors based on eigenvalues
        6. Create the projection matrix
    '''

    for i in range(IMG_NUM):
        original_imgs.append(np.array(io.imread('train/s' + str(i+1) + '.1.tif')))
        test_imgs.append(np.array(io.imread('test/s' + str(i+1) + '.2.tif').reshape([SIZE, ])))
        train_imgs.append(np.array(original_imgs[i].reshape([SIZE, ]), dtype='float64'))
        mean_img += train_imgs[i]
    else:
        mean_img /= IMG_NUM

    for i in range(IMG_NUM):
        centered_imgs.append(train_imgs[i] - mean_img)
        t += (centered_imgs[i], )

    train_mat = np.stack(t, axis=1)

    start = timeit.default_timer()
    proj_mat_trick = projection_mat_with_trick(train_mat)
    end = timeit.default_timer()
    print("Time consumtion of PCA trick algorithm is %.6fsec" %(end - start))

    start = timeit.default_timer()
    proj_mat_svd = projection_mat_with_svd(train_mat)
    end = timeit.default_timer()
    print("Time consumtion of PCA SVD algorithm is %.6fsec" % (end - start))

    projected_mat_trick = np.dot(proj_mat_trick, train_mat)
    projected_mat_svd = np.dot(proj_mat_svd, train_mat)

    print("Accuracy of result_train_trick is %.2f%%" % identify_imgs(original_imgs, mean_img, train_imgs, proj_mat_trick, projected_mat_trick, 'result_train_trick'))
    print("Accuracy of result_test_trick is %.2f%%" % identify_imgs(original_imgs, mean_img, test_imgs, proj_mat_trick, projected_mat_trick, 'result_test_trick'))
    print("Accuracy of result_train_svd is %.2f%%" % identify_imgs(original_imgs, mean_img, train_imgs, proj_mat_svd, projected_mat_svd, 'result_train_svd'))
    print("Accuracy of result_test_svd is %.2f%%" % identify_imgs(original_imgs, mean_img, test_imgs, proj_mat_svd, projected_mat_svd, 'result_test_svd'))
