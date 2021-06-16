import numpy as np
import dlib
import math
import cv2


# 面部对齐
def face_alignment(faces):
    faces_aligned = []
    for face in faces:
        predictor_path = "/Users/returnyg/PycharmProjects/MicroExpressionRecognition/additional/shape_predictor_68_face_landmarks.dat"  # dlib提供的训练好的68个人脸关键点的模型，网上可以下
        predictor = dlib.shape_predictor(predictor_path)  # 用来预测关键点
        rec = dlib.rectangle(0, 0, face.shape[0], face.shape[1])
        shape = predictor(np.uint8(face), rec)  # 注意输入的必须是uint8类型
        eye_center = ((shape.part(36).x + shape.part(45).x) * 1. / 2,  # 计算两眼的中心坐标
                      (shape.part(36).y + shape.part(45).y) * 1. / 2)
        dx = (shape.part(45).x - shape.part(36).x)  # note: right - right
        dy = (shape.part(45).y - shape.part(36).y)
        angle = math.atan2(dy, dx) * 180. / math.pi  # 计算角度
        RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)  # 计算仿射矩阵
        RotImg = cv2.warpAffine(face, RotateMatrix, (face.shape[0], face.shape[1]))  # 进行放射变换，即旋转
        faces_aligned.append(RotImg)
    return faces_aligned  # uint8
