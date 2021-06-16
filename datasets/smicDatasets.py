import os

import cv2
import numpy as np
import pandas as pd
from numpy import random
from utils.faceDealer import faceListDeal, faceListDealNew, faceListDealOF, faceListDealApex
from utils.frameNormalized import frameNormalized


# 表情类型 positive = 0; negative = 1; surprise = 2;
EMOTIONSDict = {"positive": 0, "negative": 1, "surprise": 2}
# 数据集文件位置
dataset_root = "/Users/returnyg/Datasets/SMIC/SMIC-E_raw image/HS_long/SMIC-HS-E"
datasetXls = "/Users/returnyg/Datasets/SMIC/SMIC-E_raw image/HS_long/SMIC-HS-E_annotation_2019.xlsx"


# LOO表示是否将在预处理时区分开不同受试者的样本，framenum代表取样帧数, needAlignment代表是否对齐, needFaceMask代表是否增加面部遮罩,
# needEyeMask代表是否增加眼部遮罩, OF=代表是否提取光流特征, Apexonly=代表是否仅提取处理顶点帧
def load_SMIC_data(LOO=False, framenum=30, needAlignment=False, needFaceMask=False, needEyeMask=True, OF=False, Apexonly=False):
    if os.listdir("/Users/returnyg/PycharmProjects/MicroExpressionRecognition/data_save/old").count(
            "SMICfaces.npy") > 0 and \
            os.listdir("/Users/returnyg/PycharmProjects/MicroExpressionRecognition/data_save/old").count(
                "SMICemotions.npy") > 0 and LOO is False and OF is False and Apexonly is False:
        print("Loading the SMIC data!")
        faces, emotions = readData("/Users/returnyg/PycharmProjects/MicroExpressionRecognition/data_save/old")
        return faces, emotions
    if os.listdir("/Users/returnyg/PycharmProjects/MicroExpressionRecognition/data_save/new").count(
            str(framenum)+"-SMICdata.npy") > 0 and LOO and OF is False and Apexonly is False:
        print("Loading the new SMIC data!")
        subFaceEmo = readNewData("/Users/returnyg/PycharmProjects/MicroExpressionRecognition/data_save/new", framenum=framenum)
        return subFaceEmo
    if os.listdir("/Users/returnyg/PycharmProjects/MicroExpressionRecognition/data_save/new").count("OF-SMICdata.npy") > 0 and LOO is False and OF and Apexonly is False:
        print("Loading the OF SMIC data!")
        subFaceEmo = readOFData("/Users/returnyg/PycharmProjects/MicroExpressionRecognition/data_save/new")
        return subFaceEmo
    if os.listdir("/Users/returnyg/PycharmProjects/MicroExpressionRecognition/data_save/new").count("Apex-SMICdata.npy") > 0 and LOO is False and OF is False and Apexonly:
        print("Loading the Apex SMIC data!")
        subFaceEmo = readApexData("/Users/returnyg/PycharmProjects/MicroExpressionRecognition/data_save/new")
        return subFaceEmo
    dataFolderDict = {}  # 字典形式存放数据集信息，内容为{文件名：文件路径}
    sampleList = []  # 嵌套列表形式存放样本信息，内容为[[主体编号, 样本文件夹名, 样本文件夹路径, 表情类型], []...]
    # 遍历文件夹，读取文件夹下所有受试者文件夹
    for folderName in os.listdir(dataset_root):
        if "s" in folderName and ".DS_Store" not in folderName:
            dataFolderDict[folderName] = (dataset_root + "/" + folderName)
        else:
            continue
    for folderName, folderPath in dataFolderDict.items():
        for sample in os.listdir(folderPath):
            table = pd.read_excel(datasetXls)  # table为样本信息表格文件
            subject = folderName
            innerList = []
            subject = folderName.lstrip('s')
            if int(subject) < 10:
                subject = subject.lstrip('0')
            innerList.clear()
            innerList.append(sample)
            innerList.append(folderPath + "/" + sample)
            # 查询表格获取样本情绪类型
            dataFrame = table[(table['Subject'] == int(subject)) & (table['Filename'] == sample)]
            if dataFrame.shape[0] < 1:
                # 个别样本未标记情绪类型，弃用该样本
                print(subject + ":" + sample + " Emotion is Null")
            else:
                innerList.insert(0, str(subject))
                innerList.append(EMOTIONSDict[str(dataFrame.iloc[[0], [16]].values[0][0])])
                innerList.append(dataFrame.iloc[[0], [10]].values[0][0])  # FirstFrame
                innerList.append(dataFrame.iloc[[0], [11]].values[0][0])  # LastFrame
                innerList.append(dataFrame.iloc[[0], [3]].values[0][0])  # OnsetFrame
                innerList.append(dataFrame.iloc[[0], [4]].values[0][0])  # OffsetFrame
                if not pd.isnull(dataFrame.iloc[[0], [5]].values[0][0]):
                    innerList.append(dataFrame.iloc[[0], [5]].values[0][0])  # OnsetFrame2
                    innerList.append(dataFrame.iloc[[0], [6]].values[0][0])  # OffsetFrame2
                if not pd.isnull(dataFrame.iloc[[0], [7]].values[0][0]):
                    innerList.append(dataFrame.iloc[[0], [7]].values[0][0])  # OnsetFrame3
                    innerList.append(dataFrame.iloc[[0], [8]].values[0][0])  # OffsetFrame3
                print(str(innerList[0])+" - "+str(innerList[1])+" - "+str(innerList[2])+" - "+str(innerList[3]))
                sampleList.append(innerList)
    print("样本文件夹总数量为：" + str(len(sampleList)) + " 个")
    width = 256
    height = 256
    subFaceEmo = []
    innerlist = []
    innerFaceList = []
    innerEmoList = []
    faces = []
    emotions = []
    sampleNum = sampleList[0][0]
    i = 1
    j = 0
    for sample in sampleList:
        # 从list中获取人脸的数据
        length = len(sample)
        innerface = []
        faceSubject = sample[0]
        faceFolder = sample[1]
        faceFileLoc = sample[2]
        faceEmotion = sample[3]
        firstFrame = int(sample[4])
        lastFrame = int(sample[5])
        onsetFrame = int(sample[6])
        offsetFrame = int(sample[7])
        if length > 9:
            onsetFrame2 = int(sample[8])
            offsetFrame2 = int(sample[9])
        if length > 11:
            onsetFrame3 = int(sample[10])
            offsetFrame3 = int(sample[11])
        fileList = os.listdir(faceFileLoc)
        fileList2 = []
        fileList3 = []
        print(str(i) + " / " + str(len(sampleList)))
        print(str(len(fileList)) + " : " + faceFileLoc)
        if fileList.count(".DS_Store") > 0:
            fileList.remove(".DS_Store")
        fileList.sort(key=lambda x: int(x.split('.jpg')[0].split('e')[1]), reverse=False)
        if length < 10:
            fileList = frameNormalized(fileList, onsetFrame, offsetFrame, firstFrame=firstFrame, lastFrame=lastFrame, framenum=framenum)
        elif 9 < length < 12:
            fileList, fileList2 = frameNormalized(fileList, onsetFrame, offsetFrame, firstFrame=firstFrame, lastFrame=lastFrame, onsetFrame2=onsetFrame2, offsetFrame2=offsetFrame2, framenum=framenum)
        else:
            fileList, fileList2, fileList3 = frameNormalized(fileList, onsetFrame, offsetFrame, firstFrame=firstFrame, lastFrame=lastFrame, onsetFrame2=onsetFrame2, offsetFrame2=offsetFrame2, onsetFrame3=onsetFrame3, offsetFrame3=offsetFrame3, framenum=framenum)
        if LOO and OF is False and Apexonly is False:
            if len(fileList2) < 1 and len(fileList3) < 1:
                print("正在使用新方法处理")
                if faceSubject != sampleNum:
                    innerlist.append(sampleNum)
                    innerlist.append(list(innerFaceList))
                    innerlist.append(list(innerEmoList))
                    subFaceEmo.append(list(innerlist))
                    print("当前采样人为" + str(faceSubject) + "...subFaceEmo的长度为" + str(len(subFaceEmo)) + "...采样人是" + str(
                        subFaceEmo[j][0]) + "...面部列表长度为" + str(len(subFaceEmo[j][1])) + "...情绪列表长度为" + str(
                        len(subFaceEmo[j][2])) + "..单个面部有" + str(len(subFaceEmo[j][1][0])) + "...单个表情有" + str(
                        len(subFaceEmo[j][2][0])))
                    for q in range(len(subFaceEmo)):
                        print("第{}个受试者，编号为{}，共有{}个表情样本，每个表情样本有{}个，共有{}个情绪标签，每个情绪标签有{}个。".format(
                            q + 1, subFaceEmo[q][0], len(subFaceEmo[q][1]), len(subFaceEmo[q][1][0]),
                            len(subFaceEmo[q][2]), len(subFaceEmo[q][2][0])))
                    j = j + 1
                    sampleNum = faceSubject
                    innerlist.clear()
                    innerFaceList.clear()
                    innerEmoList.clear()
                innerFaceList, innerEmoList = faceListDealNew(fileList, innerFaceList, innerEmoList, faceEmotion,
                                                              faceFileLoc, width, height, needAlignment, needFaceMask, needEyeMask, framenum=framenum)
            elif len(fileList2) > 0 and len(fileList3) < 1:
                print("正在使用新方法处理")
                if faceSubject != sampleNum:
                    innerlist.append(sampleNum)
                    innerlist.append(list(innerFaceList))
                    innerlist.append(list(innerEmoList))
                    subFaceEmo.append(list(innerlist))
                    print("当前采样人为" + str(faceSubject) + "...subFaceEmo的长度为" + str(len(subFaceEmo)) + "...采样人是" + str(
                        subFaceEmo[j][0]) + "...面部列表长度为" + str(len(subFaceEmo[j][1])) + "...情绪列表长度为" + str(
                        len(subFaceEmo[j][2])) + "..单个面部有" + str(len(subFaceEmo[j][1][0])) + "...单个表情有" + str(
                        len(subFaceEmo[j][2][0])))
                    for q in range(len(subFaceEmo)):
                        print("第{}个受试者，编号为{}，共有{}个表情样本，每个表情样本有{}个，共有{}个情绪标签，每个情绪标签有{}个。".format(
                            q + 1, subFaceEmo[q][0], len(subFaceEmo[q][1]), len(subFaceEmo[q][1][0]),
                            len(subFaceEmo[q][2]), len(subFaceEmo[q][2][0])))
                    j = j + 1
                    sampleNum = faceSubject
                    innerlist.clear()
                    innerFaceList.clear()
                    innerEmoList.clear()
                innerFaceList, innerEmoList = faceListDealNew(fileList, innerFaceList, innerEmoList, faceEmotion,
                                                              faceFileLoc, width, height, needAlignment, needFaceMask, needEyeMask, framenum=framenum)
                innerFaceList, innerEmoList = faceListDealNew(fileList2, innerFaceList, innerEmoList, faceEmotion,
                                                              faceFileLoc, width, height, needAlignment, needFaceMask, needEyeMask, framenum=framenum)
            else:
                print("正在使用新方法处理")
                if faceSubject != sampleNum:
                    innerlist.append(sampleNum)
                    innerlist.append(list(innerFaceList))
                    innerlist.append(list(innerEmoList))
                    subFaceEmo.append(list(innerlist))
                    print("当前采样人为" + str(faceSubject) + "...subFaceEmo的长度为" + str(len(subFaceEmo)) + "...采样人是" + str(
                        subFaceEmo[j][0]) + "...面部列表长度为" + str(len(subFaceEmo[j][1])) + "...情绪列表长度为" + str(
                        len(subFaceEmo[j][2])) + "..单个面部有" + str(len(subFaceEmo[j][1][0])) + "...单个表情有" + str(
                        len(subFaceEmo[j][2][0])))
                    for q in range(len(subFaceEmo)):
                        print("第{}个受试者，编号为{}，共有{}个表情样本，每个表情样本有{}个，共有{}个情绪标签，每个情绪标签有{}个。".format(
                            q + 1, subFaceEmo[q][0], len(subFaceEmo[q][1]), len(subFaceEmo[q][1][0]),
                            len(subFaceEmo[q][2]), len(subFaceEmo[q][2][0])))
                    j = j + 1
                    sampleNum = faceSubject
                    innerlist.clear()
                    innerFaceList.clear()
                    innerEmoList.clear()
                innerFaceList, innerEmoList = faceListDealNew(fileList, innerFaceList, innerEmoList, faceEmotion,
                                                              faceFileLoc, width, height, needAlignment, needFaceMask, needEyeMask, framenum=framenum)
                innerFaceList, innerEmoList = faceListDealNew(fileList2, innerFaceList, innerEmoList, faceEmotion,
                                                              faceFileLoc, width, height, needAlignment, needFaceMask, needEyeMask, framenum=framenum)
                innerFaceList, innerEmoList = faceListDealNew(fileList3, innerFaceList, innerEmoList, faceEmotion,
                                                              faceFileLoc, width, height, needAlignment, needFaceMask, needEyeMask, framenum=framenum)
        elif LOO is False and OF and Apexonly is False:
            if len(fileList2) < 1 and len(fileList3) < 1:
                print("正在使用OF方法处理")
                innerFaceList, innerEmoList = faceListDealOF(fileList, faceFileLoc, width, height, innerFaceList,
                                                             innerEmoList, faceEmotion)
                for b, lists in enumerate(innerFaceList):
                    if len(lists) < 1:
                        innerFaceList.pop(b)
                if innerFaceList and innerEmoList:
                    innerlist.append(list(innerFaceList))
                    innerlist.append(list(innerEmoList))
                if innerlist:
                    subFaceEmo.append(list(innerlist))
                innerlist.clear()
                innerFaceList.clear()
                innerEmoList.clear()
            elif len(fileList2) > 0 and len(fileList3) < 1:
                print("正在使用新方法处理")
                innerFaceList, innerEmoList = faceListDealOF(fileList, faceFileLoc, width, height, innerFaceList,
                                                             innerEmoList, faceEmotion)
                innerFaceList, innerEmoList = faceListDealOF(fileList2, faceFileLoc, width, height, innerFaceList,
                                                             innerEmoList, faceEmotion)
                for b, lists in enumerate(innerFaceList):
                    if len(lists) < 1:
                        innerFaceList.pop(b)
                if innerFaceList and innerEmoList:
                    innerlist.append(list(innerFaceList))
                    innerlist.append(list(innerEmoList))
                if innerlist:
                    subFaceEmo.append(list(innerlist))
                innerlist.clear()
                innerFaceList.clear()
                innerEmoList.clear()
            else:
                print("正在使用新方法处理")
                innerFaceList, innerEmoList = faceListDealOF(fileList, faceFileLoc, width, height, innerFaceList,
                                                             innerEmoList, faceEmotion)
                innerFaceList, innerEmoList = faceListDealOF(fileList2, faceFileLoc, width, height, innerFaceList,
                                                             innerEmoList, faceEmotion)
                innerFaceList, innerEmoList = faceListDealOF(fileList3, faceFileLoc, width, height, innerFaceList,
                                                             innerEmoList, faceEmotion)
                for b, lists in enumerate(innerFaceList):
                    if len(lists) < 1:
                        innerFaceList.pop(b)
                if innerFaceList and innerEmoList:
                    innerlist.append(list(innerFaceList))
                    innerlist.append(list(innerEmoList))
                if innerlist:
                    subFaceEmo.append(list(innerlist))
                innerlist.clear()
                innerFaceList.clear()
                innerEmoList.clear()
        elif LOO is False and OF is False and Apexonly:
            if len(fileList2) < 1 and len(fileList3) < 1:
                print("正在使用Apex方法处理")
                innerFace, innerEmo = faceListDealApex(fileList, faceFileLoc, width, height, faceEmotion, int(len(fileList) / 2))
                if isinstance(innerFace, list) and isinstance(innerEmo, list):
                    for face, emo in zip(innerFace, innerEmo):
                        innerFaceList.append(face)
                        innerEmoList.append(emo)
                else:
                    innerFaceList.append(innerFace)
                    innerEmoList.append(innerEmo)
                print(f"数据集共有{len(innerFaceList)}个表情，{len(innerEmoList)}个标签")
            elif len(fileList2) > 0 and len(fileList3) < 1:
                print("正在使用Apex方法处理")
                innerFace, innerEmo = faceListDealApex(fileList, faceFileLoc, width, height, faceEmotion, int(len(fileList) / 2))
                if isinstance(innerFace, list) and isinstance(innerEmo, list):
                    for face, emo in zip(innerFace, innerEmo):
                        innerFaceList.append(face)
                        innerEmoList.append(emo)
                else:
                    innerFaceList.append(innerFace)
                    innerEmoList.append(innerEmo)
                innerFace2, innerEmo2 = faceListDealApex(fileList2, faceFileLoc, width, height, faceEmotion, int(len(fileList) / 2))
                if isinstance(innerFace2, list) and isinstance(innerEmo2, list):
                    for face, emo in zip(innerFace2, innerEmo2):
                        innerFaceList.append(face)
                        innerEmoList.append(emo)
                else:
                    innerFaceList.append(innerFace2)
                    innerEmoList.append(innerEmo2)
                print(f"数据集共有{len(innerFaceList)}个表情，{len(innerEmoList)}个标签")
            else:
                print("正在使用Apex方法处理")
                innerFace, innerEmo = faceListDealApex(fileList, faceFileLoc, width, height, faceEmotion, int(len(fileList) / 2))
                if isinstance(innerFace, list) and isinstance(innerEmo, list):
                    for face, emo in zip(innerFace, innerEmo):
                        innerFaceList.append(face)
                        innerEmoList.append(emo)
                else:
                    innerFaceList.append(innerFace)
                    innerEmoList.append(innerEmo)
                innerFace2, innerEmo2 = faceListDealApex(fileList2, faceFileLoc, width, height, faceEmotion, int(len(fileList) / 2))
                if isinstance(innerFace2, list) and isinstance(innerEmo2, list):
                    for face, emo in zip(innerFace2, innerEmo2):
                        innerFaceList.append(face)
                        innerEmoList.append(emo)
                else:
                    innerFaceList.append(innerFace2)
                    innerEmoList.append(innerEmo2)
                innerFace3, innerEmo3 = faceListDealApex(fileList3, faceFileLoc, width, height, faceEmotion, int(len(fileList) / 2))
                if isinstance(innerFace3, list) and isinstance(innerEmo3, list):
                    for face, emo in zip(innerFace3, innerEmo3):
                        innerFaceList.append(face)
                        innerEmoList.append(emo)
                else:
                    innerFaceList.append(innerFace3)
                    innerEmoList.append(innerEmo3)
                print(f"数据集共有{len(innerFaceList)}个表情，{len(innerEmoList)}个标签")
        else:
            if len(fileList2) < 1 and len(fileList3) < 1:
                faces, emotions = faceListDeal(faces, emotions, faceEmotion, fileList, faceFileLoc, width, height)
            elif len(fileList2) > 0 and len(fileList3) < 1:
                faces, emotions = faceListDeal(faces, emotions, faceEmotion, fileList, faceFileLoc, width, height)
                faces, emotions = faceListDeal(faces, emotions, faceEmotion, fileList2, faceFileLoc, width, height)
            else:
                faces, emotions = faceListDeal(faces, emotions, faceEmotion, fileList, faceFileLoc, width, height)
                faces, emotions = faceListDeal(faces, emotions, faceEmotion, fileList2, faceFileLoc, width, height)
                faces, emotions = faceListDeal(faces, emotions, faceEmotion, fileList3, faceFileLoc, width, height)
        i = i + 1
    if innerFaceList and LOO:
        innerlist.append(sampleList[-1][0])
        innerlist.append(innerFaceList)
        innerlist.append(innerEmoList)
        subFaceEmo.append(innerlist)
        print("...subFaceEmo的长度为" + str(len(subFaceEmo)) + "...采样人是" + str(subFaceEmo[j][0]) + "...面部列表长度为" + str(
            len(subFaceEmo[j][1])) + "...情绪列表长度为" + str(
            len(subFaceEmo[j][2])))
    if LOO and OF is False and Apexonly is False:
        print("样本人总数为：" + str(len(subFaceEmo)))
        subFaceEmo = np.asarray(subFaceEmo)
        print(subFaceEmo)
        np.save('/Users/returnyg/PycharmProjects/MicroExpressionRecognition/data_save/new/'+str(framenum)+'-SMICdata.npy', subFaceEmo)
        return subFaceEmo
    elif LOO is False and OF and Apexonly is False:
        print("样本人总数为：" + str(len(subFaceEmo)))
        subFaceEmo = np.asarray(subFaceEmo)
        print(subFaceEmo)
        np.save('/Users/returnyg/PycharmProjects/MicroExpressionRecognition/data_save/new/OF-SMICdata.npy', subFaceEmo)
        return subFaceEmo
    elif LOO is False and OF is False and Apexonly:
        subFaceEmo.append(innerFaceList)
        subFaceEmo.append(innerEmoList)
        print(f"处理完毕，数据集共有{len(subFaceEmo[0])}个表情，{len(subFaceEmo[1])}个标签")
        subFaceEmo = np.asarray(subFaceEmo)
        np.save('/Users/returnyg/PycharmProjects/MicroExpressionRecognition/data_save/new/Apex-SMICdata.npy', subFaceEmo)
        return subFaceEmo
    else:
        print("smic面部" + str(len(faces)))
        print("smic情绪" + str(len(emotions)))
        faces = np.asarray(faces)
        emotions = np.asarray(emotions)
        np.save('/data_save/old/SMICfaces.npy', faces)
        np.save('/data_save/old/SMICemotions.npy', emotions)
        return faces, emotions


def input_SMIC_data():
    training_size = 265
    validation_size = 34
    test_size = 33
    all_faces, all_emotions = load_SMIC_data()
    all_faces = np.asarray(all_faces)
    randnum = random.randint(0, 100)
    random.seed(randnum)
    random.shuffle(all_faces)
    random.seed(randnum)
    random.shuffle(all_emotions)
    print("SMIC Data load success!")

    # 验证数据
    validation_faces = all_faces[training_size: training_size + validation_size]
    validation_emotions = all_emotions[training_size: training_size + validation_size]

    # 测试数据
    test_faces = all_faces[training_size + validation_size:]
    test_emotions = all_emotions[training_size + validation_size:]

    # 训练数据
    train_faces = all_faces[: training_size]
    train_emotions = all_emotions[: training_size]

    train_faces_list = []
    train_emotions_list = []
    for faces, emotion in zip(train_faces, train_emotions):
        for face in faces:
            train_faces_list.append(face)
            train_emotions_list.append(emotion)
    train_faces_list = np.expand_dims(train_faces_list, -1)

    test_faces_list = []
    test_emotions_list = []
    for faces, emotion in zip(test_faces, test_emotions):
        for face in faces:
            test_faces_list.append(face)
            test_emotions_list.append(emotion)
    test_faces_list = np.expand_dims(test_faces_list, -1)

    validation_faces_list = []
    validation_emotions_list = []
    for faces, emotion in zip(validation_faces, validation_emotions):
        for face in faces:
            validation_faces_list.append(face)
            validation_emotions_list.append(emotion)
    validation_faces_list = np.expand_dims(validation_faces_list, -1)

    return train_faces_list, train_emotions_list, test_faces_list, test_emotions_list, validation_faces_list, validation_emotions_list


def readData(filePath):
    faces = np.load(filePath + '/SMICfaces.npy', allow_pickle=True)
    emotions = np.load(filePath + '/SMICemotions.npy')
    faces = faces.tolist()
    emotions = emotions.tolist()
    return faces, emotions


def readNewData(filePath, framenum=30):
    subFaceEmo = np.load(filePath + '/' + str(framenum) + '-SMICdata.npy', allow_pickle=True)
    subFaceEmo = subFaceEmo.tolist()
    return subFaceEmo


def readOFData(filePath):
    subFaceEmo = np.load(filePath + '/' + 'OF-SMICdata.npy', allow_pickle=True)
    subFaceEmo = subFaceEmo.tolist()
    return subFaceEmo


def readApexData(filePath):
    subFaceEmo = np.load(filePath + '/' + 'Apex-SMICdata.npy', allow_pickle=True)
    subFaceEmo = subFaceEmo.tolist()
    return subFaceEmo


# t1, t2 = load_SMIC_data()
# t1 = load_SMIC_data(LOO=True, framenum=30)
# t1 = load_SMIC_data(LOO=False, framenum=100, OF=True)
t1 = load_SMIC_data(LOO=False, framenum=31, OF=False, Apexonly=True)

