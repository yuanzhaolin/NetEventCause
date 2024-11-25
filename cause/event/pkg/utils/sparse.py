import torch
import numpy as np


def softmax_average(mat, direction='row'):
    EPS = 1e-6
    result_mat = np.zeros(mat.shape)
    size = mat.shape[0] if direction == 'row' else mat.shape[1]
    for i in range(size):
        vec = mat[i, :] if direction == 'row' else mat[:, i]
        vec_tensor = torch.softmax(torch.FloatTensor(vec), dim=0)
        vec_tensor[vec_tensor < (EPS + 1.0/size)] = 0
        if vec_tensor.max() != 0:
            vec_tensor = vec_tensor / vec_tensor.sum()
        if direction == 'row':
            result_mat[i, :] = vec_tensor.numpy()
        else:
            result_mat[:, i] = vec_tensor.numpy()
        # print(L[i], vec_tensor)
    return result_mat


def find_positive(mat):
    new_mat = np.copy(mat)
    new_mat[new_mat > 1e-6] = 1
    new_mat[new_mat != 1] = 0
    return new_mat

