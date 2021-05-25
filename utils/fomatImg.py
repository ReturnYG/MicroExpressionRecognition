import cv2

# 加载opencv默认人脸识别器
CASC_PATH = "/Users/returnyg/PycharmProjects/MicroExpressionRecognition/additional/haarcascade_frontalface_default.xml"
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)


def fomat_image(image):
    # image如果为彩色图：image.shape[0][1][2](水平、垂直像素、通道数)
    if len(image.shape) > 2 and image.shape[2] == 3:
        # 将图片变为灰度图
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 它可以检测出图片中所有的人脸，并将人脸用vector保存各个人脸的坐标、大小（用矩形表示）
        # 调整scaleFactor参数的大小，可以增加识别的灵敏度，推荐1.1
        faces = cascade_classifier.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    # 如果图片中没有检测到人脸，则返回None
    if not len(faces) > 0:
        return None, None
    # max_are_face包含了人脸的坐标，大小
    max_are_face = faces[0]
    # 在所有人脸中选一张最大的脸
    for face in faces:
        if face[2] * face[3] > max_are_face[2] * max_are_face[3]:
            max_are_face = face
    # 调整图片大小，变为48*48
    try:
        image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_CUBIC)
    except Exception:
        print("problem during resize")
        return None, None
    return image
