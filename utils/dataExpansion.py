import cv2


def faceFlip(innerface):
    """
    该函数目的为增加面部镜像以扩充数据集，防止过拟合
    :param innerface: 面部数据
    :return: 镜像面部数据
    """
    innerfaceflip = []
    for img in innerface:
        faceflip = cv2.flip(img, 1)
        innerfaceflip.append(faceflip)
    return innerfaceflip


# 顶点帧水平翻转
def faceFlipApex(innerface):
    faceflip = cv2.flip(innerface, 1)
    return faceflip
