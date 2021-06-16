import cv2
import dlib
import numpy
import numpy as np


def facemask(innerface, width, height, needEyeMask=True, needFaceMask=True):
    innerfacemask = []
    predictor_path = '/Users/returnyg/PycharmProjects/MicroExpressionRecognition/additional/shape_predictor_68_face_landmarks.dat'
    predictor = dlib.shape_predictor(predictor_path)
    for face_one in innerface:
        # 寻找人脸的68个标定点
        rec = dlib.rectangle(0, 0, face_one.shape[0], face_one.shape[1])
        shape = predictor(face_one, rec)
        # 遍历所有点，打印出其坐标，并圈出来
        # for idx, pt in enumerate(shape.parts()):
        #     pt_pos = (pt.x, pt.y)
        #     cv2.circle(face_one, pt_pos, 2, (0, 0, 255), -1)  # 参数分别为：图像，圆心坐标，半径，颜色，线条粗细   # HL
            # 利用cv2.putText输出1-68
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(face_one, str(idx + 1), pt_pos, font, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
            # 位置，字体，大小，颜色，字体厚度
        landmarks = numpy.matrix([[p.x, p.y] for p in shape.parts()])
        landmarks = landmarks.tolist()
        left_eye_ppt = np.array(
            [[landmarks[36]], [landmarks[37]], [landmarks[38]], [landmarks[39]], [landmarks[40]], [landmarks[41]]],
            np.int32)
        right_eye_ppt = np.array(
            [[landmarks[42]], [landmarks[43]], [landmarks[44]], [landmarks[45]], [landmarks[46]], [landmarks[47]]],
            np.int32)
        left_eye_ppt = left_eye_ppt.reshape((-1, 1, 2))
        right_eye_ppt = right_eye_ppt.reshape((-1, 1, 2))
        if needEyeMask:
            cv2.fillPoly(face_one, [left_eye_ppt], (0, 0, 0), lineType=cv2.LINE_AA)
            cv2.fillPoly(face_one, [right_eye_ppt], (0, 0, 0), lineType=cv2.LINE_AA)
        if needFaceMask:
            cv2.rectangle(face_one, (0, (landmarks[1][1] + landmarks[2][1])//2), ((landmarks[5][0] + landmarks[6][0])//2, height), (0, 0, 0), -1, 8)
            cv2.rectangle(face_one, (width, (landmarks[14][1] + landmarks[15][1]) // 2), ((landmarks[10][0] + landmarks[11][0]) // 2, height), (0, 0, 0), -1, 8)
        innerfacemask.append(face_one)
    return innerfacemask


def facemaskApex(innerface, width, height, needEyeMask=True, needFaceMask=True):
    innerfacemask = innerface
    predictor_path = '/Users/returnyg/PycharmProjects/MicroExpressionRecognition/additional/shape_predictor_68_face_landmarks.dat'
    predictor = dlib.shape_predictor(predictor_path)
    # 寻找人脸的68个标定点
    rec = dlib.rectangle(0, 0, innerface.shape[0], innerface.shape[1])
    shape = predictor(innerface, rec)
    # 遍历所有点，打印出其坐标，并圈出来
    # for idx, pt in enumerate(shape.parts()):
    #     pt_pos = (pt.x, pt.y)
    #     cv2.circle(face_one, pt_pos, 2, (0, 0, 255), -1)  # 参数分别为：图像，圆心坐标，半径，颜色，线条粗细   # HL
        # 利用cv2.putText输出1-68
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(face_one, str(idx + 1), pt_pos, font, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
        # 位置，字体，大小，颜色，字体厚度
    landmarks = numpy.matrix([[p.x, p.y] for p in shape.parts()])
    landmarks = landmarks.tolist()
    left_eye_ppt = np.array(
        [[landmarks[36]], [landmarks[37]], [landmarks[38]], [landmarks[39]], [landmarks[40]], [landmarks[41]]],
        np.int32)
    right_eye_ppt = np.array(
        [[landmarks[42]], [landmarks[43]], [landmarks[44]], [landmarks[45]], [landmarks[46]], [landmarks[47]]],
        np.int32)
    left_eye_ppt = left_eye_ppt.reshape((-1, 1, 2))
    right_eye_ppt = right_eye_ppt.reshape((-1, 1, 2))
    if needEyeMask:
        cv2.fillPoly(innerface, [left_eye_ppt], (0, 0, 0), lineType=cv2.LINE_AA)
        cv2.fillPoly(innerface, [right_eye_ppt], (0, 0, 0), lineType=cv2.LINE_AA)
    if needFaceMask:
        cv2.rectangle(innerface, (0, (landmarks[1][1] + landmarks[2][1])//2), ((landmarks[5][0] + landmarks[6][0])//2, height), (0, 0, 0), -1, 8)
        cv2.rectangle(innerface, (width, (landmarks[14][1] + landmarks[15][1]) // 2), ((landmarks[10][0] + landmarks[11][0]) // 2, height), (0, 0, 0), -1, 8)
        innerfacemask = innerface
    return innerfacemask
