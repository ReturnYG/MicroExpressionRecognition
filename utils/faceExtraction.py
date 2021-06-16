import cv2

# 加载opencv默认人脸识别器
import numpy as np

CASC_PATH = "/Users/returnyg/PycharmProjects/MicroExpressionRecognition/additional/haarcascade_frontalface_default.xml"
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)


def compute_TVL1(prev, curr, bound):
    """Compute the TV-L1 optical flow."""
    TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
    # TVL1 = cv2.DualTVL1OpticalFlow_create()
    # TVL1 = cv2.createOptFlow_DualTVL1()
    flow = TVL1.calc(prev, curr, None)
    assert flow.dtype == np.float32

    flow = (flow + bound) * (255.0 / (2 * bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0
    return flow


def faceExtraction(fileList, faceFileLoc, width, height):
    """
    该函数目的为从图片中识别面部并剪裁，再将剪裁后的面部调整大小
    :param fileList: 包含图片文件名的文件列表
    :param faceFileLoc: 该列表中图片的路径
    :param width: 所需图片宽度
    :param height: 所需图片高度
    :return: 返回面部剪裁并修改宽高后的面部列表
    """
    innerface = []
    for image in fileList:
        image = faceFileLoc + "/" + image
        faceImage = cv2.imread(image)
        # faceImage = cv2.cvtColor(faceImage, cv2.COLOR_BGR2GRAY)
        faceLoc = cascade_classifier.detectMultiScale(faceImage, scaleFactor=1.1, minNeighbors=9)  # CASME II\SAMM 9. SMIC 15
        if type(faceLoc) == tuple:
            continue
        else:
            x, y, w, h = faceLoc[0]
            face = faceImage[y:y + h, x:x + w]
            face = cv2.resize(face, (width, height), interpolation=cv2.INTER_CUBIC)
            innerface.append(face)
    return innerface


def faceExtractionApex(fileList, faceFileLoc, width, height, apexFrameNum):
    innerface = fileList[apexFrameNum]
    image = faceFileLoc + "/" + innerface
    faceImage = cv2.imread(image)
    faceImage = cv2.cvtColor(faceImage, cv2.COLOR_BGR2GRAY)
    faceLoc = cascade_classifier.detectMultiScale(faceImage, scaleFactor=1.1, minNeighbors=9)  # CASME II\SAMM 9. SMIC 15
    if type(faceLoc) == tuple:
        pass
    else:
        x, y, w, h = faceLoc[0]
        face = faceImage[y:y + h, x:x + w]
        face = cv2.resize(face, (width, height), interpolation=cv2.INTER_CUBIC)
        innerface = face
    return innerface


def faceOFExtraction(fileList, faceFileLoc, width, height):
    innerface = []
    on = fileList[0]
    apex = fileList[int(len(fileList)/2)]
    on_img = cv2.imread(faceFileLoc + "/" + on)
    apex_img = cv2.imread(faceFileLoc + "/" + apex)
    on_gray = cv2.cvtColor(on_img, cv2.COLOR_BGR2GRAY)
    apex_gray = cv2.cvtColor(apex_img, cv2.COLOR_BGR2GRAY)

    on_faceLoc = cascade_classifier.detectMultiScale(on_gray, scaleFactor=1.1, minNeighbors=9)
    apex_faceLoc = cascade_classifier.detectMultiScale(apex_gray, scaleFactor=1.1, minNeighbors=9)  # CASME II\SAMM 9. SMIC 15

    if type(on_faceLoc) == tuple:
        pass
    else:
        x, y, w, h = on_faceLoc[0]
        face = on_gray[y:y + h, x:x + w]
        on_face = cv2.resize(face, (width, height), interpolation=cv2.INTER_CUBIC)
    if type(apex_faceLoc) == tuple:
        pass
    else:
        x, y, w, h = apex_faceLoc[0]
        face = apex_gray[y:y + h, x:x + w]
        apex_face = cv2.resize(face, (width, height), interpolation=cv2.INTER_CUBIC)

    of1 = compute_TVL1(on_face, apex_face, bound=1)
    of_img1 = cv2.addWeighted(of1[:, :, 0], 0.5, of1[:, :, 1], 0.5, 0)
    of_img1 = np.asarray(of_img1, dtype=np.uint8)

    of1 = compute_TVL1(on_face, apex_face, bound=3)
    of_img3 = cv2.addWeighted(of1[:, :, 0], 0.5, of1[:, :, 1], 0.5, 0)
    of_img3 = np.asarray(of_img3, dtype=np.uint8)

    of1 = compute_TVL1(on_face, apex_face, bound=5)
    of_img5 = cv2.addWeighted(of1[:, :, 0], 0.5, of1[:, :, 1], 0.5, 0)
    of_img5 = np.asarray(of_img5, dtype=np.uint8)

    of1 = compute_TVL1(on_face, apex_face, bound=7)
    of_img7 = cv2.addWeighted(of1[:, :, 0], 0.5, of1[:, :, 1], 0.5, 0)
    of_img7 = np.asarray(of_img7, dtype=np.uint8)

    of1 = compute_TVL1(on_face, apex_face, bound=9)
    of_img9 = cv2.addWeighted(of1[:, :, 0], 0.5, of1[:, :, 1], 0.5, 0)
    of_img9 = np.asarray(of_img9, dtype=np.uint8)

    of1 = compute_TVL1(on_face, apex_face, bound=15)
    of_img15 = cv2.addWeighted(of1[:, :, 0], 0.5, of1[:, :, 1], 0.5, 0)
    of_img15 = np.asarray(of_img15, dtype=np.uint8)

    cv2.imshow("1", of_img1)
    cv2.imshow("3", of_img3)
    cv2.imshow("5", of_img5)
    cv2.imshow("7", of_img7)
    cv2.imshow("9", of_img9)
    cv2.imshow("15", of_img15)
    cv2.waitKey(2000)
    selection = int(input("效果最好的是：\n"))
    if selection == 1:
        innerface.append(of_img1)
        return innerface
    elif selection == 3:
        innerface.append(of_img3)
        return innerface
    elif selection == 5:
        innerface.append(of_img5)
        return innerface
    elif selection == 7:
        innerface.append(of_img7)
        return innerface
    elif selection == 9:
        innerface.append(of_img9)
        return innerface
    elif selection == 15:
        innerface.append(of_img15)
        return innerface
    else:
        return innerface.clear()




