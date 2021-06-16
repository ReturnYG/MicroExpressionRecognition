import cv2

# 加载opencv默认人脸识别器
CASC_PATH = "/Users/returnyg/PycharmProjects/MicroExpressionRecognition/additional/haarcascade_frontalface_default.xml"
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)


def faceExtraction(fileList, faceFileLoc, width, height):
    innerface = []
    for image in fileList:
        image = faceFileLoc + "/" + image
        faceImage = cv2.imread(image)
        faceImage = cv2.cvtColor(faceImage, cv2.COLOR_BGR2GRAY)
        faceLoc = cascade_classifier.detectMultiScale(faceImage, scaleFactor=1.1, minNeighbors=5)
        if type(faceLoc) == tuple:
            continue
        else:
            x, y, w, h = faceLoc[0]
            face = faceImage[y:y + h, x:x + w]
            face = cv2.resize(face, (width, height), interpolation=cv2.INTER_CUBIC)
            innerface.append(face)
    return innerface
