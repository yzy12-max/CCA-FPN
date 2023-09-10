import numpy as np
import cv2
import matplotlib.pyplot as plt

# # 定义sigmoid函数
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
#
# # 定义sigmoid函数导数
# def sigmoid_derivative(x):
#     return sigmoid(x) * (1 - sigmoid(x))
#
# # 生成x值
# x = np.linspace(-10, 10, 100)
#
# # 绘制sigmoid函数及其导数
# plt.plot(x, sigmoid(x), label='sigmoid')
# plt.plot(x, sigmoid_derivative(x), label='derivative')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Sigmoid Function and Its Derivative')
# plt.legend()
# plt.grid(False)
# plt.show()

# # 定义softmax函数
# def softmax(x):
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum()
#
# # 定义softmax函数导数
# def softmax_derivative(x):
#     s = softmax(x).reshape(-1,1)
#     return np.diagflat(s) - np.dot(s, s.T)
#
# # 生成x值
# x = np.array([1, 2, 3, 4, 5])
#
# # 绘制softmax函数及其导数
# plt.plot(x, softmax(x), label='softmax')
# plt.plot(x, softmax_derivative(x), label='derivative')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Softmax Function and Its Derivative')
# plt.legend()
# plt.grid(True)
# plt.show()

# # 定义ReLU函数
# def ReLU(x):
#     return np.maximum(0,x)
#
# # 定义ReLU函数导数
# def ReLU_derivative(x):
#     return np.where(x > 0, 1, 0)
#
# # 生成x值
# x = np.linspace(-1, 1, 100)
#
# # 绘制ReLU函数及其导数
# plt.plot(x, ReLU(x), label='ReLU')
# plt.plot(x, ReLU_derivative(x), label='derivative')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('ReLU Function and Its Derivative')
# plt.legend()
# plt.grid(False)
# plt.show()

# # 定义Leaky ReLU函数
# def Leaky_ReLU(x, alpha=0.1):
#     return np.where(x > 0, x, alpha * x)
#
# # 定义Leaky ReLU函数导数
# def Leaky_ReLU_derivative(x, alpha=0.1):
#     return np.where(x > 0, 1, alpha)
#
# # 生成x值
# x = np.linspace(-1, 1, 100)
#
# # 绘制Leaky ReLU函数及其导数
# plt.plot(x, Leaky_ReLU(x), label='Leaky ReLU')
# plt.plot(x, Leaky_ReLU_derivative(x), label='derivative')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Leaky ReLU Function and Its Derivative')
# plt.legend()
# plt.grid(False)
# plt.show()

# # 定义SiLU函数
# def SiLU(x):
#     return x / (1 + np.exp(-x))
#
# # 定义SiLU函数导数
# def SiLU_derivative(x):
#     return (1 + np.exp(-x) + x * np.exp(-x)) / ((1 + np.exp(-x)) ** 2)
#
# # 生成x值
# x = np.linspace(-10, 10, 100)
#
# # 绘制SiLU函数及其导数
# plt.plot(x, SiLU(x), label='SiLU')
# plt.plot(x, SiLU_derivative(x), label='derivative')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('SiLU Function and Its Derivative')
# plt.legend()
# plt.grid(False)
# plt.show()


# def smooth_L1_loss(x, y):
#     diff = np.abs(x - y)
#     if diff < 1:
#         return 0.5 * diff ** 2
#     else:
#         return diff - 0.5
#
# def smooth_L1_loss_derivative(x, y):
#     diff = np.abs(x - y)
#     if diff < 1:
#         return x - y
#     else:
#         return 1
#
# # 定义x和y的取值范围
# x_range = np.arange(-5, 5, 0.1)
# y_range = np.arange(-5, 5, 0.1)
#
# # 创建网格点坐标矩阵
# X, Y = np.meshgrid(x_range, y_range)
#
# # 计算每个坐标点的损失值和导数值
# Z_loss = np.zeros_like(X)
# Z_derivative = np.zeros_like(X)
# for i in range(X.shape[0]):
#     for j in range(X.shape[1]):
#         Z_loss[i, j] = smooth_L1_loss(X[i, j], Y[i, j])
#         Z_derivative[i, j] = smooth_L1_loss_derivative(X[i, j], Y[i, j])
#
# # 绘制损失函数图像
# plt.figure(figsize=(8, 6))
# plt.title('Smooth L1 Loss')
# plt.xlabel('x')
# plt.ylabel('Loss')
# plt.grid()
# plt.contourf(X, Y, Z_loss, levels=50, cmap='jet')
# plt.colorbar()
# plt.show()
#
# # 绘制导数图像
# plt.figure(figsize=(8, 6))
# plt.title('Smooth L1 Loss Derivative')
# plt.xlabel('x')
# plt.ylabel('Derivative')
# plt.grid()
# plt.contourf(X, Y, Z_derivative, levels=50, cmap='jet')
# plt.colorbar()
# plt.show()

# def smooth_L1_loss(x, y):
#     diff = np.abs(x - y)
#     if diff < 1:
#         return 0.5 * diff ** 2
#     else:
#         return diff - 0.5
#
# def smooth_L1_loss_derivative(x, y):
#     diff = np.abs(x - y)
#     if diff < 1:
#         return x - y
#     else:
#         return 1
#
# # 定义x和y的取值范围
# x_range = np.arange(-5, 5, 0.1)
#
# # 计算每个坐标点的损失值和导数值
# y_loss = np.zeros_like(x_range)
# y_derivative = np.zeros_like(x_range)
# for i in range(len(x_range)):
#     y_loss[i] = smooth_L1_loss(x_range[i], 0)
#     y_derivative[i] = smooth_L1_loss_derivative(x_range[i], 0)
#
# # 绘制损失函数和导数曲线图
# fig, ax = plt.subplots(figsize=(8, 6))
#
# ax.set_title('Smooth L1 Loss and Derivative')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.plot(x_range, y_loss, label='Loss')
# ax.plot(x_range, y_derivative, label='Derivative')
# ax.legend()
# ax.grid(False)
# plt.show()
#
# pass










# # 创建一个
# img = cv2.imread('/data1/yzycode/mmdetection/demo/demo.jpg')
# # cv2.namedWindow('img',cv2.WINDOW_NORMAL)  # 窗口
# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # coding: utf-8 https://www.mianshigee.com/note/detail/55701ufy/
def motion_blur(image, degree=24, angle=45):
  image = np.array(image)
  # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
  M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)  # 第一个参数旋转中心，第二个参数旋转角度，第三个参数：缩放比例
  motion_blur_kernel = np.diag(np.ones(degree)) # 对角取1
  motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree)) # 得到矩阵后得用到图像的仿射变换函数才可以进行最终图像的变化
  motion_blur_kernel = motion_blur_kernel / degree
  blurred = cv2.filter2D(image, -1, motion_blur_kernel)
  # convert to uint8
  cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
  blurred = np.array(blurred, dtype=np.uint8)
  return blurred
img = cv2.imread('/data1/yzycode/mmdetection/demo/000003.jpg')
img_ = motion_blur(img)
# cv2.imshow('Source image',img)
# cv2.waitKey(0)
# cv2.imshow('blur image',img_)
# cv2.waitKey(0)
cv2.imwrite('/data1/yzycode/mmdetection/demo/000003img_.png',img_)
pass


