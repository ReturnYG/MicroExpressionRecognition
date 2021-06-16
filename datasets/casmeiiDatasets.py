import os
import numpy as np
import pandas as pd
from numpy import random
from utils.dataExpansion import faceFlip
from utils.faceExtraction import faceExtraction
from utils.frameNormalized import frameNormalized

# 表情类型 disgust = 1; sadness = 1; surprise = 2; repression = 1; happiness = 0; fear  = 1; others = 3;
# happiness => positive; disgust, sadness, fear, repression => negative; surprise => surprise; others => others
EMOTIONSDict = {"disgust": 1, "sadness": 1, "surprise": 2, "repression": 1, "happiness": 0, "fear": 1, "others": 3}
# 数据集文件位置
dataset_root = ["/Users/returnyg/Datasets/CASME2/CASME2-RAW"]
datasetXls = ["/Users/returnyg/Datasets/CASME2/CASME2-coding-20190701.xlsx"]


def load_CASMEII_data():
    if os.listdir("/Users/returnyg/PycharmProjects/MicroExpressionRecognition/data_save").count(
            "CASMEIIfaces.npy") > 0 and \
            os.listdir("/Users/returnyg/PycharmProjects/MicroExpressionRecognition/data_save").count(
                "CASMEIIemotions.npy") > 0:
        print("Loading the CASMEII data!")
        faces, emotions = readData("/Users/returnyg/PycharmProjects/MicroExpressionRecognition/data_save")
        return faces, emotions
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
                innerList.insert(0, str(subject))
                innerList.append(EMOTIONSDict[str(dataFrame.iloc[[0], [8]].values[0][0])])  # Estimated Emotion
                innerList.append(dataFrame.iloc[[0], [3]].values[0][0])  # OnsetFrame
                innerList.append(dataFrame.iloc[[0], [4]].values[0][0])  # ApexFrame
                innerList.append(dataFrame.iloc[[0], [5]].values[0][0])  # OffsetFrame
                print(str(innerList[0]) + " - " + str(innerList[1]) + " - " + str(innerList[2]) + " - " + str(
                    innerList[3]) + " - " + str(innerList[4]) + " - " + str(innerList[5]) + " - " + str(innerList[6]))
                sampleList.append(innerList)
    print("样本总体数量为：" + str(len(sampleList)) + " 个")
    width = 224
    height = 224
    faces = []
    emotions = []
    i = 1
    for sample in sampleList:
        # 从list中获取人脸的数据
        innerface = []
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
        fileList = frameNormalized(fileList, onsetFrame, offsetFrame, apexFrame)
        print(fileList)
        innerface = faceExtraction(fileList, faceFileLoc, width, height)
        innerfaceflip = faceFlip(innerface)
        faces.append(innerface)
        emotions.append(faceEmotion)
        faces.append(innerfaceflip)
        emotions.append(faceEmotion)
        i = i + 1
    print("casmeii面部" + str(len(faces)))
    print("casmeii情绪" + str(len(emotions)))
    faces = np.asarray(faces)
    emotions = np.asarray(emotions)
    np.save('/Users/returnyg/PycharmProjects/MicroExpressionRecognition/data_save/CASMEIIfaces.npy', faces)
    np.save('/Users/returnyg/PycharmProjects/MicroExpressionRecognition/data_save/CASMEIIemotions.npy', emotions)
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
