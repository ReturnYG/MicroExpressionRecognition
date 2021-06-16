import os
import numpy as np
import pandas as pd
from numpy import random
from utils.dataExpansion import faceFlip
from utils.faceExtraction import faceExtraction
from utils.frameNormalized import frameNormalized


# 表情类型 positive = 0; negative = 1; surprise = 2;
EMOTIONSDict = {"positive": 0, "negative": 1, "surprise": 2}
# 数据集文件位置
dataset_root = "/Users/returnyg/Datasets/SMIC/SMIC-E_raw image/HS_long/SMIC-HS-E"
datasetXls = "/Users/returnyg/Datasets/SMIC/SMIC-E_raw image/HS_long/SMIC-HS-E_annotation_2019.xlsx"


def load_SMIC_data():
    if os.listdir("/Users/returnyg/PycharmProjects/MicroExpressionRecognition/data_save").count(
            "SMICfaces.npy") > 0 and \
            os.listdir("/Users/returnyg/PycharmProjects/MicroExpressionRecognition/data_save").count(
                "SMICemotions.npy") > 0:
        print("Loading the SMIC data!")
        faces, emotions = readData("/Users/returnyg/PycharmProjects/MicroExpressionRecognition/data_save")
        return faces, emotions
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
    width = 224
    height = 224
    faces = []
    emotions = []
    i = 1
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
            fileList = frameNormalized(fileList, onsetFrame, offsetFrame, firstFrame=firstFrame, lastFrame=lastFrame)
        elif 9 < length < 12:
            fileList, fileList2 = frameNormalized(fileList, onsetFrame, offsetFrame, firstFrame=firstFrame, lastFrame=lastFrame, onsetFrame2=onsetFrame2, offsetFrame2=offsetFrame2)
        else:
            fileList, fileList2, fileList3 = frameNormalized(fileList, onsetFrame, offsetFrame, firstFrame=firstFrame, lastFrame=lastFrame, onsetFrame2=onsetFrame2, offsetFrame2=offsetFrame2, onsetFrame3=onsetFrame3, offsetFrame3=offsetFrame3)
        if len(fileList2) < 1 and len(fileList3) < 1:
            innerface = faceExtraction(fileList, faceFileLoc, width, height)
            innerfaceflip = faceFlip(innerface)
            faces.append(innerface)
            emotions.append(faceEmotion)
            faces.append(innerfaceflip)
            emotions.append(faceEmotion)
        elif len(fileList2) > 0 and len(fileList3) < 1:
            innerface = faceExtraction(fileList, faceFileLoc, width, height)
            faces.append(innerface)
            emotions.append(faceEmotion)
            innerface = faceExtraction(fileList2, faceFileLoc, width, height)
            innerfaceflip = faceFlip(innerface)
            faces.append(innerface)
            emotions.append(faceEmotion)
            faces.append(innerfaceflip)
            emotions.append(faceEmotion)
        else:
            innerface = faceExtraction(fileList, faceFileLoc, width, height)
            innerfaceflip = faceFlip(innerface)
            faces.append(innerface)
            emotions.append(faceEmotion)
            faces.append(innerfaceflip)
            emotions.append(faceEmotion)
            innerface = faceExtraction(fileList2, faceFileLoc, width, height)
            innerfaceflip = faceFlip(innerface)
            faces.append(innerface)
            emotions.append(faceEmotion)
            faces.append(innerfaceflip)
            emotions.append(faceEmotion)
            innerface = faceExtraction(fileList3, faceFileLoc, width, height)
            innerfaceflip = faceFlip(innerface)
            faces.append(innerface)
            emotions.append(faceEmotion)
            faces.append(innerfaceflip)
            emotions.append(faceEmotion)
        print(len(faces))
        i = i + 1
    print("smic面部" + str(len(faces)))
    print("smic情绪" + str(len(emotions)))
    faces = np.asarray(faces)
    emotions = np.asarray(emotions)
    np.save('/Users/returnyg/PycharmProjects/MicroExpressionRecognition/data_save/SMICfaces.npy', faces)
    np.save('/Users/returnyg/PycharmProjects/MicroExpressionRecognition/data_save/SMICemotions.npy', emotions)
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


t1, t2 = load_SMIC_data()


