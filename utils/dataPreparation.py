import random
import tensorflow as tf
import numpy as np
from utils.batchSizeFix import batchFuc
from utils.sampleBalance import sample_balance


# 将储存数据的列表降为一维
def flatten(input_list):
    output_list = []
    while True:
        if not input_list:
            break
        for index, i in enumerate(input_list):
            if type(i) == list:
                input_list = i + input_list[index + 1:]
                break
            else:
                output_list.append(i)
                input_list.pop(index)
                break
    return output_list


# 训练前的数据准备
def dataPraparation(dataset, epoch, framenum, batch_size, LOO=True, OF=False):
    # 如果采用LOSO训练方式，则迭代次数为受试者人数，每次使用一个受试者为验证集，其余受试者为训练集
    if LOO:
        testset = [dataset[epoch % len(dataset)]]
        trainset = list(dataset)
        trainset.pop(epoch % len(dataset))
        testfaces = []
        testemotions = []
        trainfaces = []
        trainemotions = []
        print("测试样本为{}号".format(testset[0][0]))
        for subject in testset:
            testfaces.append(subject[1])
            testemotions.append(subject[2])
        for subject in trainset:
            trainfaces.append(subject[1])
            trainemotions.append(subject[2])

        trainfaceslist = flatten(trainfaces)
        trainemotionslist = flatten(trainemotions)
        testfaceslist = flatten(testfaces)
        testemotionslist = flatten(testemotions)

        trainfaceslist_afterNorm = []
        tetfaceslist_afterNorm = []
        # 图像归一化
        for face in trainfaceslist:
            newFace = (face - np.min(face)) / (np.max(face) - np.min(face))
            trainfaceslist_afterNorm.append(newFace)
        for face in testfaceslist:
            newFace = (face - np.min(face)) / (np.max(face) - np.min(face))
            tetfaceslist_afterNorm.append(newFace)

        # 划分batch_size
        groupnum = framenum
        trainemotionslist_withGroup = batchFuc(trainemotionslist, groupnum)
        testemotionslist_withGroup = batchFuc(testemotionslist, groupnum)
        trainfaceslist_withGroup = batchFuc(trainfaceslist_afterNorm, groupnum)
        testfaceslist_withGroup = batchFuc(tetfaceslist_afterNorm, groupnum)

        trainfaceslist_withBatch_balanced, trainemotionslist_withBatch_balanced = sample_balance(trainfaceslist_withGroup, trainemotionslist_withGroup)

        trainfaceslist_withGroup_flatten = flatten(trainfaceslist_withBatch_balanced)
        trainemotionslist_withGroup_flatten = flatten(trainemotionslist_withBatch_balanced)
        testfaceslist_withGroup_flatten = flatten(testfaceslist_withGroup)
        testemotionslist_withGroup_flatten = flatten(testemotionslist_withGroup)

        trainemotionslist_withBatch = batchFuc(trainemotionslist_withGroup_flatten, batch_size)
        testemotionslist_withBatch = batchFuc(testemotionslist_withGroup_flatten, batch_size)
        trainfaceslist_withBatch = batchFuc(trainfaceslist_withGroup_flatten, batch_size)
        testfaceslist_withBatch = batchFuc(testfaceslist_withGroup_flatten, batch_size)

        ziplist = list(zip(trainfaceslist_withBatch, trainemotionslist_withBatch))
        random.shuffle(ziplist)  # 打乱训练集
        trainfaceslist_withBatch_shuffle, trainemotionslist_withBatch_shuffle = zip(*ziplist)

        # one-hot编码
        trainEmotionList = []
        for i in trainemotionslist_withBatch_shuffle:
            trainEmotionList.append(tf.one_hot(i, depth=3))
        testEmotionList = []
        for i in testemotionslist_withBatch:
            testEmotionList.append(tf.one_hot(i, depth=3))
        # 增加通道数维度
        trainFaceList = []
        for i in trainfaceslist_withBatch_shuffle:
            trainFaceList.append(np.expand_dims(i, -1))  # np.expand_dims(i, -1)
        testFaceList = []
        for i in testfaceslist_withBatch:
            testFaceList.append(np.expand_dims(i, -1))  # np.expand_dims(i, -1)
        return trainFaceList, trainEmotionList, testFaceList, testEmotionList
    # 如果训练光流特征，则随机添加20%数据为验证集，其余为训练集
    elif OF:
        testset = []
        for loc, i in enumerate(dataset):
            j = random.randint(1, 10)
            if j > 8:
                testset.append(i)
                dataset.pop(loc)
        trainset = list(dataset)
        testfaces = []
        testemotions = []
        trainfaces = []
        trainemotions = []
        for subject in testset:
            testfaces.append(subject[0])
            testemotions.append(subject[1])
        for subject in trainset:
            trainfaces.append(subject[0])
            trainemotions.append(subject[1])
        trainfaceslist = flatten(trainfaces)
        trainemotionslist = flatten(trainemotions)
        testfaceslist = flatten(testfaces)
        testemotionslist = flatten(testemotions)
        trainfaceslist_afterNorm = []
        tetfaceslist_afterNorm = []
        # 图像归一化
        for face in trainfaceslist:
            newFace = (face - np.min(face)) / (np.max(face) - np.min(face))
            trainfaceslist_afterNorm.append(newFace)
        for face in testfaceslist:
            newFace = (face - np.min(face)) / (np.max(face) - np.min(face))
            tetfaceslist_afterNorm.append(newFace)
        trainemotionslist_withGroup = batchFuc(trainemotionslist, batch_size)
        testemotionslist_withGroup = batchFuc(testemotionslist, batch_size)
        trainfaceslist_withGroup = batchFuc(trainfaceslist_afterNorm, batch_size)
        testfaceslist_withGroup = batchFuc(tetfaceslist_afterNorm, batch_size)
        # one-hot编码
        trainEmotionList = []
        for i in trainemotionslist_withGroup:
            trainEmotionList.append(tf.one_hot(i, depth=3))
        testEmotionList = []
        for i in testemotionslist_withGroup:
            testEmotionList.append(tf.one_hot(i, depth=3))
        # 增加通道数维度
        trainFaceList = []
        for i in trainfaceslist_withGroup:
            trainFaceList.append(np.expand_dims(i, -1))  # np.expand_dims(i, -1)
        testFaceList = []
        for i in testfaceslist_withGroup:
            testFaceList.append(np.expand_dims(i, -1))  # np.expand_dims(i, -1)
        return trainFaceList, trainEmotionList, testFaceList, testEmotionList

