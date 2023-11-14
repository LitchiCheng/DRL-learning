import numpy as np

# 定义概率转移矩阵
P = np.array([[0.9, 0.1],
              [0.5, 0.5]])

# 初始状态
state = np.array([1, 0])

# 迭代次数
n_iterations = 10

# 马尔科夫链迭代
for _ in range(n_iterations):
    state = np.dot(state, P)
    print(state)
