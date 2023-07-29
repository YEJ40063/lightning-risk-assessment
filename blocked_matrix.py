import numpy as np
import cupy as cp
from scipy import sparse
import tracemalloc
import linecache
import os
import sys
from weakref import ref
from tqdm import tqdm

block_size = 3000
dtype = 'float64'

def block_model_f(model, data):
    if data.shape[0]>5000:
        modelout = np.zeros((data.shape[0], 2))
        for i in tqdm(range(0, data.shape[0], block_size)):
            part_out = model(data[i:i + block_size, :])
            modelout[i:i + block_size, :] = part_out
    return modelout

def display_top(snapshot, key_type='lineno', limit=2):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#%s: %s:%s: %.1f KiB"
              % (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))
def dot(A, B, block_size=2000, numpy=True):
    # 创建结果矩阵 C
    C = cp.zeros((A.shape[0], B.shape[1]), dtype=dtype)
    # C = sparse.lil_matrix((A.shape[0], B.shape[1]), dtype=np.float64)
    if A.shape[1] > block_size:
        for i in range(0, A.shape[0], block_size):
            for j in range(0, B.shape[1], block_size):
                for k in range(0, A.shape[1], block_size):
                    A_block = cp.asarray(A[i:i + block_size, k: k + block_size])
                    B_block = cp.asarray(B[k: k + block_size, j:j + block_size])
                    # 执行逐元素点乘运算
                    C_block = cp.dot(A_block, B_block)
                    del A_block
                    del B_block
                    # 将子矩阵块的点乘结果放置在正确的位置
                    # C[i:i + block_size, j:j + block_size] += cp.asnumpy(C_block)
                    C[i:i + block_size, j:j + block_size] += C_block
                    del C_block
    else:
        for i in range(0, A.shape[0], block_size):
            for j in range(0, B.shape[1], block_size):
                A_block = cp.array(A[i:i+block_size, :])
                B_block = cp.array(B[:, j:j+block_size])
                # 执行逐元素点乘运算
                C_block = cp.dot(A_block, B_block)
                # 将子矩阵块的点乘结果放置在正确的位置
                C[i:i+block_size, j:j+block_size] = C_block
    if numpy:
        return cp.asnumpy(C)
    else:
        return C

def add(A, B, block_size=block_size, numpy=True):
    # 创建结果矩阵 C
    C = cp.zeros((A.shape[0], B.shape[1]), dtype=dtype)
    # C = sparse.lil_matrix((A.shape[0], B.shape[1]), dtype=np.float64)
    for i in range(0, A.shape[0], block_size):
        for j in range(0, B.shape[1], block_size):
            A_block = cp.array(A[i:i+block_size, j:j+block_size])
            B_block = cp.array(B[i:i+block_size, j:j+block_size])
            # 执行逐元素点乘运算
            C_block = A_block + B_block
            # 将子矩阵块的点乘结果放置在正确的位置
            C[i:i+block_size, j:j+block_size] = C_block
    if numpy:
        return cp.asnumpy(C)
    else:
        return C

def sqrt(A, block_size=block_size, numpy=True):
    # 创建结果矩阵 C
    C = cp.zeros((A.shape[0], A.shape[1]), dtype=dtype)
    # C = sparse.lil_matrix((A.shape[0], B.shape[1]), dtype=np.float64)
    for i in range(0, A.shape[0], block_size):
        for j in range(0, A.shape[1], block_size):
            A_block = cp.array(A[i:i+block_size, j:j+block_size])

            C_block = cp.sqrt(A_block)
            # 将子矩阵块的点乘结果放置在正确的位置
            C[i:i+block_size, j:j+block_size] = C_block
    if numpy:
        return cp.asnumpy(C)
    else:
        return C

def nan_to_num(A, block_size=block_size, numpy=True):
    # 创建结果矩阵 C
    C = cp.zeros((A.shape[0], A.shape[1]), dtype=dtype)
    # C = sparse.lil_matrix((A.shape[0], B.shape[1]), dtype=np.float64)
    for i in range(0, A.shape[0], block_size):
        for j in range(0, A.shape[1], block_size):
            A_block = cp.array(A[i:i+block_size, j:j+block_size])

            C_block = cp.nan_to_num(A_block)
            # 将子矩阵块的点乘结果放置在正确的位置
            C[i:i+block_size, j:j+block_size] = C_block
    if numpy:
        return cp.asnumpy(C)
    else:
        return C
def sub(A, B, block_size=block_size, numpy=True):
    # 创建结果矩阵 C
    C = cp.zeros((A.shape[0], B.shape[1]), dtype=dtype)
    # C = sparse.lil_matrix((A.shape[0], B.shape[1]), dtype=np.float64)
    for i in range(0, A.shape[0], block_size):
        for j in range(0, B.shape[1], block_size):
            A_block = cp.array(A[i:i+block_size, j:j+block_size])
            B_block = cp.array(B[i:i+block_size, j:j+block_size])
            # 执行逐元素点乘运算
            C_block = A_block - B_block
            # 将子矩阵块的点乘结果放置在正确的位置
            C[i:i+block_size, j:j+block_size] = C_block
    if numpy:
        return cp.asnumpy(C)
    else:
        return C

def exp(A, block_size=block_size, numpy=True, sparse=False):
    # 创建结果矩阵 C
    C = cp.zeros((A.shape[0], A.shape[1]), dtype=dtype)
    # C = sparse.lil_matrix((A.shape[0], B.shape[1]), dtype=np.float64)
    if sparse:
        for i in range(0, A.shape[0], block_size):
            for j in range(0, A.shape[1], block_size):
                A_block = cp.array(A[i:i + block_size, j:j + block_size])
                row, col = A_block.nonzero()
                # 将子矩阵块的点乘结果放置在正确的位置
                C[row, col] = cp.exp(A_block[row, col])
    else:
        for i in range(0, A.shape[0], block_size):
            for j in range(0, A.shape[1], block_size):
                A_block = cp.array(A[i:i+block_size, j:j+block_size])

                C_block = cp.exp(A_block)
                # 将子矩阵块的点乘结果放置在正确的位置
                C[i:i+block_size, j:j+block_size] = C_block
    if numpy:
        return cp.asnumpy(C)
    else:
        return C

def mul(A, B, block_size=block_size, numpy=True):
    # 创建结果矩阵 C
    C = cp.zeros((B.shape[0], B.shape[1]), dtype=dtype)
    # C = sparse.lil_matrix((A.shape[0], B.shape[1]), dtype=np.float64)
    if isinstance(A, np.ndarray):
        for i in range(0, B.shape[0], block_size):
            for j in range(0, B.shape[1], block_size):
                A_block = cp.array(A[i:i+block_size, j:j+block_size])
                B_block = cp.array(B[i:i+block_size, j:j+block_size])
                # 执行逐元素点乘运算
                C_block = A_block * B_block
                # 将子矩阵块的点乘结果放置在正确的位置
                C[i:i+block_size, j:j+block_size] = C_block
    else:
        for i in range(0, B.shape[0], block_size):
            for j in range(0, B.shape[1], block_size):
                B_block = cp.array(B[i:i+block_size, j:j+block_size])
                # 执行逐元素点乘运算
                C_block = A * B_block
                # 将子矩阵块的点乘结果放置在正确的位置
                C[i:i+block_size, j:j+block_size] = C_block
    if numpy:
        return cp.asnumpy(C)
    else:
        return C

def squ(A, block_size=block_size, numpy=True):
    # 创建结果矩阵 C
    C = cp.zeros((A.shape[0], A.shape[1]), dtype=dtype)
    # C = sparse.lil_matrix((A.shape[0], B.shape[1]), dtype=np.float64)
    for i in range(0, A.shape[0], block_size):
        for j in range(0, A.shape[1], block_size):
            A_block = cp.array(A[i:i+block_size, j:j+block_size])

            C_block = A_block ** 2
            # 将子矩阵块的点乘结果放置在正确的位置
            C[i:i+block_size, j:j+block_size] = C_block
    if numpy:
        return cp.asnumpy(C)
    else:
        return C

def div(A, k, block_size=block_size, numpy=True):
    # 创建结果矩阵 C
    C = cp.zeros((A.shape[0], A.shape[1]), dtype=dtype)
    # C = sparse.lil_matrix((A.shape[0], B.shape[1]), dtype=np.float64)
    for i in range(0, A.shape[0], block_size):
        for j in range(0, A.shape[1], block_size):
            A_block = cp.array(A[i:i+block_size, j:j+block_size])

            C_block = A_block / k
            # 将子矩阵块的点乘结果放置在正确的位置
            C[i:i+block_size, j:j+block_size] = C_block
    if numpy:
        return cp.asnumpy(C)
    else:
        return C
def argsort(A, k=5, block_size=block_size, numpy=True):
    # 创建结果矩阵 C
    C = cp.zeros((A.shape[0], int(k)), dtype='int32')
    # C = sparse.lil_matrix((A.shape[0], B.shape[1]), dtype=np.float64)
    for i in range(0, A.shape[0], block_size):
        A_block = cp.array(A[i:i+block_size, :])

        C_block = A_block.argsort()[:, 1:k+1]
        # 将子矩阵块的点乘结果放置在正确的位置
        C[i:i+block_size, :] = C_block
    if numpy:
        return cp.asnumpy(C)
    else:
        return C

# def normaolize(A, max_A, min_A, block_size=block_size):
#     # 创建结果矩阵 C
#     C = cp.zeros((A.shape[0], A.shape[1]), dtype=dtype)
#     # C = sparse.lil_matrix((A.shape[0], B.shape[1]), dtype=np.float64)
#     for i in range(0, A.shape[0], block_size):
#         for j in range(0, A.shape[1], block_size):
#             A_block = cp.array(A[i:i + block_size, j:j + block_size])
#             max_A
#
#             C_block = A_block ** 2
#             # 将子矩阵块的点乘结果放置在正确的位置
#             C[i:i + block_size, j:j + block_size] = C_block
#     return cp.asnumpy(C)

if __name__ == '__main__':
    from sklearn.datasets import make_moons
    np.random.seed(5)
    X, Y = make_moons(n_samples=200, noise=0.05)
    ind_0 = np.nonzero(Y == 0)[0]
    ind_1 = np.nonzero(Y == 1)[0]
    Y[ind_0] = -1

    ind_l0 = np.random.choice(ind_0, 1, False)
    ind_u0 = np.setdiff1d(ind_0, ind_l0)
    ind_l1 = np.random.choice(ind_1, 1, False)
    ind_u1 = np.setdiff1d(ind_1, ind_l1)

    Xl = np.vstack([X[ind_l0, :], X[ind_l1, :]])
    Yl = np.hstack([Y[ind_l0], Y[ind_l1]])
    Xu = np.vstack([X[ind_u0, :], X[ind_u1, :]])
    np.random.seed(1)
    A = np.random.randn(1000, 1000)
    np.random.seed(2)
    B = np.random.randn(1000, 1000)
    C = dot(Xu, Xu.T)
    D = Xu.dot(Xu.T)
    print((C-D).sum())