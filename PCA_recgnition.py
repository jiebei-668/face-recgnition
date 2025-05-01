#!/usr/bin/env python
# coding: utf-8
import pickle

# In[3]:


import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt

IMAGE_SIZE = (50, 50)

PCA_NUM = 5
# 1、加载训练集中的脸，转为一个M行N列矩阵T
def load_data(path):
    # 查看路径下所有文件
    train_file_name = os.listdir(path)
    train_file_path_list = list()
    for one in train_file_name:
        train_file_path_list.append(os.path.join(path, one))
    # 计算有几个文件（图片命名都是以 序号.jpg方式）减去Thumbs.db
    # file_number = len(train_file_path) - 1
    T = []
    # 把所有图片转为1-D并存入T中
    for i, one_train_file_path in enumerate(train_file_path_list):
        image = cv.imread(one_train_file_path, cv.IMREAD_GRAYSCALE)
        image = cv.resize(image, IMAGE_SIZE)
        # 转为1-D
        image = image.reshape(image.size, 1)
        T.append(image)
    T = np.array(T)
    # 不能直接T.reshape(T.shape[1],T.shape[0]) 这样会打乱顺序，
    T = T.reshape(T.shape[0], T.shape[1])
    # 输出的是 2500, train_nums 维度数据
    return np.mat(T).T, train_file_path_list

def PCA(T):
    # 2、对T进行0均值化
    # 把均值变为0 axis = 1代表对各行求均值
    m = T.mean(axis=1)
    A = T - m


    # 计算 A.T * A 的特征向量和特征值, 其中 V 是特征值，D 是特征向量
    L = (A.T) * (A)
    V, D = np.linalg.eig(L)
    idx = V.argsort()[::-1]
    V = V[idx]
    D = D[:,idx]

    # 计算 A *AT的特征向量
    eigenface = A * D
    eigenface = eigenface[:,:PCA_NUM]
    return eigenface, m, A

def showEigenfaces(eigenface, num_eigenfaces):
    fig, axes = plt.subplots(1, num_eigenfaces, figsize=(15, 5))
    for i in range(num_eigenfaces):
        # 确保绘制的图像是实数
        axes[i].imshow(np.real(eigenface[:, i].reshape(IMAGE_SIZE)), cmap='gray')
        axes[i].axis('off')
    plt.show()

# 图片的命名为 cls-name.jpg 4-2025010101010101000.jpg
def recognize(testImage, eigenface, m, A):
    # 记录测试图片的实际类别和预测类别
    actual_cls = testImage.split("/")[-1].split("-")[0]
    _, trainNumber = np.shape(A)
    # 4、计算投影后的矩阵P
    # 投影到特征脸后的
    projectedImage = eigenface.T * (A)
    # 可解决中文路径不能打开问题
    testImageArray = cv.imdecode(np.fromfile(testImage, dtype=np.uint8), cv.IMREAD_GRAYSCALE)
    # 转为1-D
    testImageArray = cv.resize(testImageArray, IMAGE_SIZE)
    testImageArray = testImageArray.reshape(testImageArray.size, 1)
    testImageArray = np.mat(np.array(testImageArray))
    differenceTestImage = testImageArray - m

    projectedTestImage = eigenface.T * (differenceTestImage)
    distance = []
    # 6、计算test_P和P中每个样本的距离，选出最近的那个即可
    for i in range(0, trainNumber):
        q = projectedImage[:, i]
        temp = np.linalg.norm(projectedTestImage - q)
        distance.append(temp)

    minDistance = min(distance)
    index = distance.index(minDistance)
    print('similar index is ', index + 1)
    cv.imshow("test data", cv.imread(testImage, cv.IMREAD_GRAYSCALE))

    cv.imshow("recognize result", cv.imread('./TrainDatabase' + '/' + str(index + 1) + '.jpg', cv.IMREAD_GRAYSCALE))
    cv.waitKey()
    return index + 1


# 预测单张图片，testImage是测试图片的路径（推荐绝对路径），eigenface，m和A是pca返回的结果，train_file_path[i]对应A[:, i]
# 返回预测成功与否
# mode="train"返回的是 if_right(判断预测是否正确) 和 predict_cls
# mode ="test"返回的是 true和 predict_cls
def predict(testImage, eigenface, m, A, train_file_path, mode="train"):
    # 记录测试图片的实际类别和预测类别
    actual_cls = testImage.split("/")[-1].split("-")[0]
    _, trainNumber = np.shape(A)
    # 4、计算投影后的矩阵P
    # 投影到特征脸后的
    projectedImage = eigenface.T * (A)
    # 可解决中文路径不能打开问题
    testImageArray = cv.imdecode(np.fromfile(testImage, dtype=np.uint8), cv.IMREAD_GRAYSCALE)
    # 转为1-D
    testImageArray = cv.resize(testImageArray, IMAGE_SIZE)
    testImageArray = testImageArray.reshape(testImageArray.size, 1)
    testImageArray = np.mat(np.array(testImageArray))
    differenceTestImage = testImageArray - m

    projectedTestImage = eigenface.T * (differenceTestImage)
    distance_map = dict()
    distance = []
    # 6、计算test_P和P中每个样本的距离，选出最近的那个即可
    # 初始化最小距离和索引
    min_distance = float('inf')  # 用正无穷作为初始最小值
    min_index = -1  # 初始索引设为无效值

    # 假设在循环中计算距离（例如遍历训练样本）
    for i in range(trainNumber):  # trainNumber是训练样本总数
        # 假设distance是当前计算的距离（需替换为实际计算代码）
        q = projectedImage[:, i]
        distance = np.linalg.norm(projectedTestImage - q)

        # 动态更新最小值和索引
        if distance < min_distance:
            min_distance = distance
            min_index = i  # 记录当前循环的索引

    # 获取预测的结果图片的路径
    pre_path = train_file_path[min_index]
    predicted_cls = pre_path.split("/")[-1].split("-")[0]

    # cv.imshow("test data", cv.imread(testImage, cv.IMREAD_GRAYSCALE))

    # cv.imshow("recognize result", cv.imread(pre_path,  cv.IMREAD_GRAYSCALE))
    # cv.waitKey()
    return predicted_cls == actual_cls if mode == "train" else True, predicted_cls

# 使用训练数据训练，训练好后进行测试，并将pca训练得到的 eigenface，m，A，和train_filepath_list都保存
def train_and_save(test_dir="TestDatabase", save_path="paramDir"):
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    # 获取训练数据，默认所有类别图片默认放在 TrainDatabase 目录下
    # T形状为 2500, trainnums， train_filepath_list[i]对应T[:, i]
    T, train_filepath_list = load_data(os.path.abspath('TrainDatabase'))
    # 使用PCA得到数据特征
    eigenface, m, A = PCA(T)
    with open(os.path.join(save_path, "train_filepath_list.pkl"), "wb") as f:
        pickle.dump(train_filepath_list, f)
    with open(os.path.join(save_path, "eigenface.pkl"), "wb") as f:
        pickle.dump(train_filepath_list, f)
    with open(os.path.join(save_path, "m.pkl"), "wb") as f:
        pickle.dump(train_filepath_list, f)
    with open(os.path.join(save_path, "A.pkl"), "wb") as f:
        pickle.dump(train_filepath_list, f)
    # showEigenfaces(eigenface, PCA_NUM)  # 显示特征脸
    # eigenface是提取的特征
    test_filename_list = os.listdir(test_dir)
    test_filepath_list = list()
    for one in test_filename_list:
        test_filepath_list.append(os.path.join(test_dir, one))
    total_test_nums = len(test_filepath_list)
    right_nums = 0
    for ii, one_testfilepath in enumerate(test_filepath_list):
        pre_result, predicted_cls = predict(os.path.abspath(one_testfilepath), eigenface, m, A, train_filepath_list, mode="train")
        right_nums += 1 if pre_result == True else 0
    print("total test sample nums: ", total_test_nums)
    print("acc:  ", right_nums / total_test_nums)


if __name__ == '__main__':
    train_and_save()


# In[ ]:




