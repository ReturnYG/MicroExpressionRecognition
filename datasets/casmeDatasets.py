import os
import cv2
import numpy as np
import pandas as pd
from numpy import random

# 表情类型 disgust = 1; sadness = 2; surprise = 3; repression = 4; happiness = 5; tense  =6; fear = 7; contempt = 8;
EMOTIONSDict = {"disgust": 1, "sadness": 2, "surprise": 3, "repression": 4, "happiness": 5, "tense": 6, "fear": 7,
                "comtempt": 8}
# 数据集文件位置
datasetSection_root = ["/Users/returnyg/Datasets/CASME/RAW/CASME_A/Section A",
                       "/Users/returnyg/Datasets/CASME/RAW/CASME_B/Section B"]
datasetXls = ["/Users/returnyg/Datasets/CASME/RAW/CASME_A/Section A/Section A.xls",
              "/Users/returnyg/Datasets/CASME/RAW/CASME_B/Section B/Section B.xls"]
# 加载opencv默认人脸识别器
CASC_PATH = "/Users/returnyg/PycharmProjects/MicroExpressionRecognition/additional/haarcascade_frontalface_default.xml"
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)


def load_CASME_data():
    if os.listdir("/Users/returnyg/PycharmProjects/MicroExpressionRecognition/data_save").count("CASMEfaces.npy") > 0 and \
            os.listdir("/Users/returnyg/PycharmProjects/MicroExpressionRecognition/data_save").count("CASMEemotions.npy") > 0:
        faces, emotions = readData("/Users/returnyg/PycharmProjects/MicroExpressionRecognition/data_save")
        return faces, emotions
    dataFolderDict = {}  # 字典形式存放数据集信息，内容为{文件名：文件路径}
    sampleList = []  # 嵌套列表形式存放样本信息，内容为[[主体编号, 样本文件夹名, 样本文件夹路径, 表情类型], []...]
    # 遍历文件夹，读取文件夹下所有受试者文件夹
    for dataSetRoot in datasetSection_root:
        for folderName in os.listdir(dataSetRoot):
            if "sub" in folderName and ".DS_Store" not in folderName:
                dataFolderDict[folderName] = (dataSetRoot + "/" + folderName)
            else:
                continue

    for folderName, folderPath in dataFolderDict.items():
        for sample in os.listdir(folderPath):
            tableA = pd.read_excel(datasetXls[0], sheet_name='Sheet1')  # tableA\tableB为样本信息表格文件
            tableB = pd.read_excel(datasetXls[1], sheet_name='Sheet1')
            subject = folderName
            innerList = []

            if '0' in subject:
                subject = folderName.lstrip('sub0')
                innerList.clear()
                innerList.append(sample)
                innerList.append(folderPath + "/" + sample)
                if int(subject) < 8:
                    # 查询表格获取样本情绪类型
                    dataFrame = tableA[(tableA['Subject'] == int(subject)) & (tableA['Filename'] == sample)]
                    if dataFrame.shape[0] < 1:
                        # 个别样本未标记情绪类型，弃用该样本
                        print(subject + ":" + sample + "Emotion is Null")
                    else:
                        innerList.insert(0, str(subject))
                        innerList.append(EMOTIONSDict[str(dataFrame.iloc[[0], [11]].values[0][0])])
                        sampleList.append(innerList)
                else:
                    dataFrame = tableB[(tableB['Subject'] == int(subject)) & (tableB['Filename'] == sample)]
                    if dataFrame.shape[0] < 1:
                        print(subject + ":" + sample + "Emotion is Null")
                    else:
                        innerList.insert(0, str(subject))
                        innerList.append(EMOTIONSDict[str(dataFrame.iloc[[0], [11]].values[0][0])])
                        sampleList.append(innerList)
            else:
                subject = folderName.lstrip('sub')
                innerList.clear()
                innerList.append(sample)
                innerList.append(folderPath + "/" + sample)
                if int(subject) < 8:
                    dataFrame = tableA[(tableA['Subject'] == int(subject)) & (tableA['Filename'] == sample)]
                    if dataFrame.shape[0] < 1:
                        print(subject + ":" + sample + " Emotion is Null")
                    else:
                        innerList.insert(0, str(subject))
                        innerList.append(EMOTIONSDict[str(dataFrame.iloc[[0], [11]].values[0][0])])
                        sampleList.append(innerList)
                else:
                    dataFrame = tableB[(tableB['Subject'] == int(subject)) & (tableB['Filename'] == sample)]
                    if dataFrame.shape[0] < 1:
                        print(subject + ":" + sample + " Emotion is Null")
                    else:
                        innerList.insert(0, str(subject))
                        innerList.append(EMOTIONSDict[str(dataFrame.iloc[[0], [11]].values[0][0])])
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
        fileList = os.listdir(faceFileLoc)
        print(str(i) + " / " + str(len(sampleList)))
        print(str(len(fileList)) + ":" + faceFileLoc)
        if fileList.count(".DS_Store") > 0:
            fileList.remove(".DS_Store")
        print(fileList)
        if "-" in fileList[0]:
            fileList.sort(key=lambda x: int(x.replace(faceFolder, '').replace('.jpg', '')), reverse=True)
        else:
            fileList.sort(key=lambda x: int(x.split('img')[1].split('.')[0]), reverse=True)
        for image in fileList:
            image = faceFileLoc + "/" + image
            # 把脸的数据变为224*224像素
            faceImage = cv2.imread(image)
            faceImage = cv2.cvtColor(faceImage, cv2.COLOR_BGR2GRAY)
            face = cascade_classifier.detectMultiScale(faceImage, scaleFactor=1.1, minNeighbors=5)
            if type(face) == tuple:
                continue
            else:
                face = np.asarray(face).astype("uint8")
                face = cv2.resize(face, (width, height), interpolation=cv2.INTER_CUBIC)
                innerface.append(face)
        faces.append(innerface)
        emotions.append(faceEmotion)
        i = i + 1
    # 把faces从列表变为三维矩阵。(35887,)----->(35887,48,48)
    print(len(faces))
    print(len(emotions))
    faces = np.asarray(faces)
    emotions = np.asarray(emotions)
    np.save('/Users/returnyg/PycharmProjects/MicroExpressionRecognition/data_save/CASMEfaces.npy', faces)
    np.save('/Users/returnyg/PycharmProjects/MicroExpressionRecognition/data_save/CASMEemotions.npy', emotions)
    return faces, emotions


def input_CASME_data():
    training_size = 142
    validation_size = 18
    test_size = 17
    all_faces, all_emotions = load_CASME_data()
    all_faces = np.asarray(all_faces)
    randnum = random.randint(0, 100)
    random.seed(randnum)
    random.shuffle(all_faces)
    random.seed(randnum)
    random.shuffle(all_emotions)
    print("CASME Data load success!")

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

    return train_faces_list, train_emotions_list, test_faces_list, test_emotions_list


def readData(filePath):
    faces = np.load(filePath+'/CASMEfaces.npy', allow_pickle=True)
    emotions = np.load(filePath+'/CASMEemotions.npy')
    faces = faces.tolist()
    emotions = emotions.tolist()
    return faces, emotions

