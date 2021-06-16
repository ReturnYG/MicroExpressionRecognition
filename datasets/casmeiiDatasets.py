import os

import cv2
import numpy as np
import pandas as pd
from numpy import random
from utils.dataExpansion import faceFlip
from utils.faceDealer import faceListDeal, faceListDealNew, faceListDealOF, faceListDealApex
from utils.faceExtraction import faceExtraction
from utils.frameNormalized import frameNormalized

# 表情类型 disgust = 1; sadness = 1; surprise = 2; repression = 3; happiness = 0; fear  = 1; others = 3;
# happiness => positive; disgust, sadness, fear => negative; surprise => surprise; others， repression => others
EMOTIONSDict = {"disgust": 1, "sadness": 1, "surprise": 2, "repression": 3, "happiness": 0, "fear": 1, "others": 3}
# 数据集文件位置
dataset_root = ["/Users/returnyg/Datasets/CASME2/CASME2-RAW"]
datasetXls = ["/Users/returnyg/Datasets/CASME2/CASME2-coding-20190701.xlsx"]


# LOO表示是否将在预处理时区分开不同受试者的样本，framenum代表取样帧数, needAlignment代表是否对齐, needFaceMask代表是否增加面部遮罩,
# needEyeMask代表是否增加眼部遮罩, OF=代表是否提取光流特征, Apexonly=代表是否仅提取处理顶点帧
def load_CASMEII_data(LOO=False, framenum=30, needAlignment=False, needFaceMask=False, needEyeMask=True, OF=False, Apexonly=False):
    if os.listdir("/Users/returnyg/PycharmProjects/MicroExpressionRecognition/data_save/old").count(
            "CASMEIIfaces.npy") > 0 and os.listdir("/Users/returnyg/PycharmProjects/MicroExpressionRecognition/data_save/old").count(
                "CASMEIIemotions.npy") > 0 and LOO is False and OF is False and Apexonly is False:
        print("Loading the CASMEII data!")
        faces, emotions = readData("/Users/returnyg/PycharmProjects/MicroExpressionRecognition/data_save")
        return faces, emotions
    if os.listdir("/Users/returnyg/PycharmProjects/MicroExpressionRecognition/data_save/new").count(
            str(framenum)+"-CASMEIIdata.npy") > 0 and LOO and OF is False and Apexonly is False:
        print("Loading the new CASMEII data!")
        subFaceEmo = readNewData("/Users/returnyg/PycharmProjects/MicroExpressionRecognition/data_save/new", framenum=framenum)
        return subFaceEmo
    if os.listdir("/Users/returnyg/PycharmProjects/MicroExpressionRecognition/data_save/new").count("OF-CASMEIIdata.npy") > 0 and LOO is False and OF and Apexonly is False:
        print("Loading the OF CASMEII data!")
        subFaceEmo = readOFData("/Users/returnyg/PycharmProjects/MicroExpressionRecognition/data_save/new")
        return subFaceEmo
    if os.listdir("/Users/returnyg/PycharmProjects/MicroExpressionRecognition/data_save/new").count("Apex-CASMEIIdata.npy") > 0 and LOO is False and OF is False and Apexonly:
        print("Loading the Apex CASMEII data!")
        subFaceEmo = readApexData("/Users/returnyg/PycharmProjects/MicroExpressionRecognition/data_save/new")
        return subFaceEmo
    dataFolderDict = {}  # 字典形式存放数据集信息，内容为{文件名：文件路径}
    sampleList = []  # 嵌套列表形式存放样本信息，内容为[[主体编号, 样本文件夹名, 样本文件夹路径, 表情类型], []...]
    # 遍历文件夹，读取文件夹下所有受试者文件夹
    for folderName in os.listdir(dataset_root[0]):
        if "sub" in folderName and ".DS_Store" not in folderName:
            dataFolderDict[folderName] = (dataset_root[0] + "/" + folderName)
        else:
            continue
    for folderName, folderPath in dataFolderDict.items():
        for sample in os.listdir(folderPath):
            table = pd.read_excel(datasetXls[0], sheet_name='Sheet1')  # table为样本信息表格文件
            subject = folderName
            innerList = []
            subject = folderName.lstrip('sub')
            innerList.clear()
            innerList.append(sample)
            innerList.append(folderPath + "/" + sample)
            # 查询表格获取样本情绪类型
            dataFrame = table[(table['Subject'] == int(subject)) & (table['Filename'] == sample)]
            if dataFrame.shape[0] < 1:
                # 个别样本未标记情绪类型，弃用该样本
                print(subject + ":" + sample + " Emotion is Null")
            else:
                if str(dataFrame.iloc[[0], [8]].values[0][0]) == 'others':
                    print("其他样本不算入内")
                    continue
                if str(dataFrame.iloc[[0], [8]].values[0][0]) == 'repression':
                    print("repression样本不算入内")
                    continue
                innerList.insert(0, str(subject))
                innerList.append(EMOTIONSDict[str(dataFrame.iloc[[0], [8]].values[0][0])])  # Estimated Emotion
                innerList.append(dataFrame.iloc[[0], [3]].values[0][0])  # OnsetFrame
                innerList.append(dataFrame.iloc[[0], [4]].values[0][0])  # ApexFrame
                innerList.append(dataFrame.iloc[[0], [5]].values[0][0])  # OffsetFrame
                print(str(innerList[0]) + " - " + str(innerList[1]) + " - " + str(innerList[2]) + " - " + str(
                    innerList[3]) + " - " + str(innerList[4]) + " - " + str(innerList[5]) + " - " + str(innerList[6]))
                sampleList.append(innerList)
    print("样本总体数量为：" + str(len(sampleList)) + " 个")
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
        faceSubject = sample[0]
        faceFolder = sample[1]
        faceFileLoc = sample[2]
        faceEmotion = sample[3]
        onsetFrame = sample[4]
        apexFrame = sample[5]
        offsetFrame = sample[6]
        fileList = os.listdir(faceFileLoc)
        print(str(i) + " / " + str(len(sampleList)))
        print(str(len(fileList)) + " : " + faceFileLoc)
        if fileList.count(".DS_Store") > 0:
            fileList.remove(".DS_Store")
        fileList.sort(key=lambda x: int(x.split('img')[1].split('.')[0]), reverse=False)
        print(fileList)
        fileList = frameNormalized(fileList, onsetFrame, offsetFrame, apexFrame, framenum=framenum)
        print(fileList)
        if LOO and OF is False and Apexonly is False:
            print("正在使用新方法处理")
            if faceSubject != sampleNum:
                innerlist.append(sampleNum)
                innerlist.append(list(innerFaceList))
                innerlist.append(list(innerEmoList))
                subFaceEmo.append(list(innerlist))
                for q in range(len(subFaceEmo)):
                    print("第{}个受试者，编号为{}，共有{}个表情样本，每个表情样本有{}个，共有{}个情绪标签，每个情绪标签有{}个。".format(
                        q+1, subFaceEmo[q][0], len(subFaceEmo[q][1]), len(subFaceEmo[q][1][0]), len(subFaceEmo[q][2]), len(subFaceEmo[q][2][0])))
                    for j in range(len(subFaceEmo[q][1])):
                        print("这是第{}个受试者的第{}个表情，有{}个，情绪标签有{}个".format(q + 1, j + 1, len(subFaceEmo[q][1][j]), len(subFaceEmo[q][2][j])))
                j = j + 1
                sampleNum = faceSubject
                innerlist.clear()
                innerFaceList.clear()
                innerEmoList.clear()
            innerFaceList, innerEmoList = faceListDealNew(fileList, innerFaceList, innerEmoList, faceEmotion, faceFileLoc, width, height, needAlignment, needFaceMask, needEyeMask, framenum=framenum)
        elif LOO is False and OF and Apexonly is False:
            print("正在使用OF方法处理")
            innerFaceList, innerEmoList = faceListDealOF(fileList, faceFileLoc, width, height, innerFaceList, innerEmoList, faceEmotion)
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
            print("正在使用Apex方法处理")
            innerFace, innerEmo = faceListDealApex(fileList, faceFileLoc, width, height, faceEmotion, int(len(fileList)/2))
            if isinstance(innerFace, list) and isinstance(innerEmo, list):
                for face, emo in zip(innerFace, innerEmo):
                    innerFaceList.append(face)
                    innerEmoList.append(emo)
            else:
                innerFaceList.append(innerFace)
                innerEmoList.append(innerEmo)
            print(f"数据集共有{len(innerFaceList)}个表情，{len(innerEmoList)}个标签")
        elif LOO is False and OF is False and Apexonly is False:
            faces, emotions = faceListDeal(faces, emotions, faceEmotion, fileList, faceFileLoc, width, height)
        i = i + 1
        print("######################################################")
    if innerFaceList and LOO and OF is False and Apexonly is False:
        innerlist.append(sampleList[-1][0])
        innerlist.append(innerFaceList)
        innerlist.append(innerEmoList)
        subFaceEmo.append(innerlist)
        print("...subFaceEmo的长度为" + str(len(subFaceEmo)) + "...采样人是" + str(subFaceEmo[j][0]) + "...面部列表长度为" + str(len(subFaceEmo[j][1])) + "...情绪列表长度为" + str(
            len(subFaceEmo[j][2])))
    if LOO and OF is False and Apexonly is False:
        print("样本人总数为：" + str(len(subFaceEmo)))
        subFaceEmo = np.asarray(subFaceEmo)
        np.save('/Users/returnyg/PycharmProjects/MicroExpressionRecognition/data_save/new/'+str(framenum)+'-CASMEIIdata.npy', subFaceEmo)
        return subFaceEmo
    if LOO is False and OF and Apexonly is False:
        print("样本人总数为：" + str(len(subFaceEmo)))
        subFaceEmo = np.asarray(subFaceEmo)
        np.save('/Users/returnyg/PycharmProjects/MicroExpressionRecognition/data_save/new/OF-CASMEIIdata.npy', subFaceEmo)
        return subFaceEmo
    if LOO is False and OF is False and Apexonly:
        subFaceEmo.append(innerFaceList)
        subFaceEmo.append(innerEmoList)
        print(f"处理完毕，数据集共有{len(subFaceEmo[0])}个表情，{len(subFaceEmo[1])}个标签")
        subFaceEmo = np.asarray(subFaceEmo)
        np.save('/Users/returnyg/PycharmProjects/MicroExpressionRecognition/data_save/new/Apex-CASMEIIdata.npy', subFaceEmo)
        return subFaceEmo
    else:
        print("casmeii面部" + str(len(faces)))
        print("casmeii情绪" + str(len(emotions)))
        faces = np.asarray(faces)
        emotions = np.asarray(emotions)
        np.save('/Users/returnyg/PycharmProjects/MicroExpressionRecognition/data_save/old/CASMEIIfaces.npy', faces)
        np.save('/Users/returnyg/PycharmProjects/MicroExpressionRecognition/data_save/old/CASMEIIemotions.npy', emotions)
        return faces, emotions


def input_CASMEII_data():
    training_size = 408
    validation_size = 51
    test_size = 51
    all_faces, all_emotions = load_CASMEII_data()
    all_faces = np.asarray(all_faces)
    randnum = random.randint(0, 100)
    random.seed(randnum)
    random.shuffle(all_faces)
    random.seed(randnum)
    random.shuffle(all_emotions)
    print("CASMEII Data load success!")

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
    faces = np.load(filePath + '/CASMEIIfaces.npy', allow_pickle=True)
    emotions = np.load(filePath + '/CASMEIIemotions.npy')
    faces = faces.tolist()
    emotions = emotions.tolist()
    return faces, emotions


def readNewData(filePath, framenum=30):
    subFaceEmo = np.load(filePath + '/' + str(framenum) + '-CASMEIIdata.npy', allow_pickle=True)
    subFaceEmo = subFaceEmo.tolist()
    return subFaceEmo


def readOFData(filePath):
    subFaceEmo = np.load(filePath + '/' + 'OF-CASMEIIdata.npy', allow_pickle=True)
    subFaceEmo = subFaceEmo.tolist()
    return subFaceEmo


def readApexData(filePath):
    subFaceEmo = np.load(filePath + '/' + 'Apex-CASMEIIdata.npy', allow_pickle=True)
    subFaceEmo = subFaceEmo.tolist()
    return subFaceEmo


# t1 = load_CASMEII_data(LOO=True, framenum=30)
# t1 = load_CASMEII_data(LOO=False, framenum=200, OF=True)
t1 = load_CASMEII_data(LOO=False, framenum=31, OF=False, Apexonly=True)
