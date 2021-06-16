import cv2


def faceFlip(innerface):
    innerfaceflip = []
    for img in innerface:
        faceflip = cv2.flip(img, 1)
        innerfaceflip.append(faceflip)
    return innerfaceflip
