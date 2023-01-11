import numpy as np
from numpy.linalg import multi_dot, eig


def gen_adjmat(worker_num,nei):  # nei_num doesn't include code itself.
    adj_mat = np.zeros((worker_num, worker_num))
    for i in range(worker_num):
        for j in range(int(nei / 2)):
            adj_mat[i][(i - (j + 1)) % worker_num] = 1
            adj_mat[i][(i + (j + 1)) % worker_num] = 1
            adj_mat[i][i] = 0
   
    return adj_mat

def my_inv(mat):
    threshold = 1e-3
    e_vals, e_vecs = eig(mat)
    e_vals = np.real(e_vals)
    e_vecs = np.real(e_vecs)
    inv_e_vals = np.where(e_vals > threshold, 1 / e_vals, e_vals)
    inv_Sigma = np.diag(inv_e_vals)
    inv_mat = multi_dot([e_vecs, inv_Sigma, e_vecs.T])

    return inv_mat



if __name__ == "__main__":
    adj_mat = gen_adjmat(20,2)
    #print(adj_mat)