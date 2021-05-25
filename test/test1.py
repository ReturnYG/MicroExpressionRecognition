import os
import cv2
import numpy as np

CASC_PATH = "/Users/returnyg/PycharmProjects/MicroExpressionRecognition/additional/haarcascade_frontalface_default.xml"
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)


def test_load_data():
    faces = []
    emotions = []
    fileloc = ["/Users/returnyg/Datasets/CASME/RAW/CASME_A/Section A/sub07/EP01_6",
               "/Users/returnyg/Datasets/CASME/RAW/CASME_A/Section A/sub03/EP10",
               "/Users/returnyg/Datasets/CASME/RAW/CASME_A/Section A/sub03/EP09_4",
               "/Users/returnyg/Datasets/CASME/RAW/CASME_A/Section A/sub03/EP09_3"]
    filename = ["EP01_6", "EP10", "EP09_4", "EP09_3"]
    j = 0
    for fileLoc, i in zip(fileloc, filename):
        innerface = []
        filelist = os.listdir(fileLoc)
        if filelist.count(".DS_Store") > 0:
            filelist.remove(".DS_Store")
        print(filelist)
        if "-" in filelist[0]:
            str = i + '-'
            filelist.sort(key=lambda x: int(x.replace(str, '').replace('.jpg', '')), reverse=True)
        else:
            filelist.sort(key=lambda x: int(x.split('img')[1].split('.')[0]), reverse=True)
        for image in filelist:
            image = fileLoc + "/" + image
            # 把脸的数据变为224*224像素
            faceImage = cv2.imread(image)
            faceImage = cv2.cvtColor(faceImage, cv2.COLOR_BGR2GRAY)
            face = cascade_classifier.detectMultiScale(faceImage, scaleFactor=1.1, minNeighbors=5)
            if type(face) == tuple:
                continue
            else:
                face = np.asarray(face).astype("uint8")
                face = cv2.resize(face, (224, 224), interpolation=cv2.INTER_CUBIC)
                innerface.append(face)
        faces.append(innerface)
        emotions.append(j)
        j = j + 1
    # 把faces从列表变为三维矩阵。(35887,)----->(35887,48,48)
    faces = np.asarray(faces)
    print(len(faces))
    print(len(emotions))

    train_faces_list = []
    train_emotions_list = []
    for faces, emotion in zip(faces, emotions):
        for face in faces:
            train_faces_list.append(face)
            train_emotions_list.append(emotion)
    print(len(train_faces_list))
    print(len(train_emotions_list))
    train_faces_list = np.expand_dims(train_faces_list, -1)
    return train_faces_list, train_emotions_list
