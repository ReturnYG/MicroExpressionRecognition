import cv2

from utils.dataExpansion import faceFlip, faceFlipApex
from utils.faceAlignment import face_alignment
from utils.faceExtraction import faceExtraction, faceOFExtraction, faceExtractionApex
from utils.facemask import facemask, facemaskApex


def faceListDeal(faces, emotions, faceEmotion, fileList, faceFileLoc, width, height):
    innerface = faceExtraction(fileList, faceFileLoc, width, height)
    innerfaceWithAlignment = face_alignment(innerface)
    innerfaceflipWithAlignment = faceFlip(innerfaceWithAlignment)
    innerfaceWithFacemask = facemask(innerfaceWithAlignment, width, height)
    innerfaceflipWithFacemask = facemask(innerfaceflipWithAlignment, width, height)
    faces.append(innerfaceWithFacemask)
    emotions.append(faceEmotion)
    faces.append(innerfaceflipWithFacemask)
    emotions.append(faceEmotion)
    return faces, emotions


def faceListDealNew(fileList, innerFaceList, innerEmoList, faceEmotion, faceFileLoc, width, height, needAlignment=True, needFaceMask=True, needEyeMask=True, framenum=30):
    emotions = []
    emotions.extend([faceEmotion] * framenum)
    innerface = faceExtraction(fileList, faceFileLoc, width, height)
    if needAlignment:
        innerfaceWithAlignment = face_alignment(innerface)
    else:
        innerfaceWithAlignment = innerface
    if faceEmotion == 1:
        innerfaceWithFacemask = facemask(innerfaceWithAlignment, width, height, needEyeMask, needFaceMask)
        innerFaceList.append(innerfaceWithFacemask)
        innerEmoList.append(emotions)
        return innerFaceList, innerEmoList
    else:
        innerfaceflipWithAlignment = faceFlip(innerfaceWithAlignment)
        innerfaceWithFacemask = facemask(innerfaceWithAlignment, width, height, needEyeMask, needFaceMask)
        innerfaceflipWithFacemask = facemask(innerfaceflipWithAlignment, width, height, needEyeMask, needFaceMask)
        innerFaceList.append(innerfaceWithFacemask)
        innerFaceList.append(innerfaceflipWithFacemask)
        innerEmoList.append(emotions)
        innerEmoList.append(emotions)
        return innerFaceList, innerEmoList


def faceListDealOF(fileList, faceFileLoc, width, height, innerFaceList, innerEmoList, faceEmotion):
    emotions = []
    emotions.extend([faceEmotion] * 1)
    innerface = faceOFExtraction(fileList, faceFileLoc, width, height)
    if isinstance(innerface, list):
        innerFaceList.append(innerface)
        innerEmoList.append(emotions)
        return innerFaceList, innerEmoList
    else:
        return innerFaceList, innerEmoList


def faceListDealApex(fileList, faceFileLoc, width, height, faceEmotion, apexFrameNum):
    if faceEmotion == 1:
        emotions = faceEmotion
        innerface = faceExtractionApex(fileList, faceFileLoc, width, height, apexFrameNum)
        innerfaceWithMask = facemaskApex(innerface, width, height, needEyeMask=True, needFaceMask=False)
        return innerfaceWithMask, emotions
    else:
        emotions = [faceEmotion, faceEmotion]
        innerface = faceExtractionApex(fileList, faceFileLoc, width, height, apexFrameNum)
        innerfaceWithMask = facemaskApex(innerface, width, height, needEyeMask=True, needFaceMask=False)
        innerfaceWithMaskFlip = faceFlipApex(innerfaceWithMask)
        return [innerfaceWithMask, innerfaceWithMaskFlip], emotions


def faceImgCheck(faceFileLoc, CheckLoc, faces):
    print("###############Check-START################")
    print(f"当前检查的图像文件路径为：{faceFileLoc}")
    print(f"当前检查的图像处理步骤批为：{CheckLoc}")
    print(f"共有{len(faces)}张图片")
    for i in faces:
        cv2.imshow(CheckLoc, i)
        cv2.waitKey(0)
        cv2.destroyWindow(CheckLoc)
    print("###############Check-FINISH###############")
