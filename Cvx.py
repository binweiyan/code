import cvxpy as cp
import numpy as np

# 定义问题的维度
n = 2  # x 的维度
m = 3  # 绝对值项的数量

# 定义二次项的系数 Q (对称且半正定)
Q = np.array([[2, 0], [0, 2]])

# 定义一次项的系数 c
c = np.array([1, -1])

# 定义绝对值项的系数 a_i 和 b_i
A = np.array([[1, 2], [-1, 1], [0, -1]])  # a_i 矩阵，每一行是 a_i 向量
b = np.array([1, -2, 0])  # b_i 向量

# 定义优化变量
x = cp.Variable(n)

# 目标函数中的二次项和线性项
objective = 0.5 * cp.quad_form(x, Q) + c.T @ x

# 绝对值项
abs_terms = cp.sum(cp.abs(A @ x + b))

# 总目标函数
objective += abs_terms

# 定义优化问题
problem = cp.Problem(cp.Minimize(objective))

# 求解问题
problem.solve()

# 输出结果
print("最优解 x:", x.value)
print("最小化目标值:", problem.value)
