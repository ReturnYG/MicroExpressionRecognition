def frameNormalized(fileList, onsetFrame, offsetFrame, apexFrame=None, firstFrame=None, lastFrame=None,
                    onsetFrame2=None, offsetFrame2=None, onsetFrame3=None, offsetFrame3=None, framenum=30):
    """
    该函数目的为将表情长度归一为30
    :param fileList: 图片文件名列表
    :param onsetFrame: 从表格读取的表情第一帧
    :param offsetFrame: 从表格读取的表情最后一帧
    :param apexFrame: 从表格读取的表情顶点帧
    :param firstFrame: 完整图片序列第一张
    :param lastFrame: 完整图片序列最后一张
    :param onsetFrame2: 从表格读取的第二个表情第一帧
    :param offsetFrame2: 从表格读取的第二个表情最后一帧
    :param onsetFrame3: 从表格读取的第三个表情第一帧
    :param offsetFrame3: 从表格读取的第三个表情最后一帧
    :return: 返回固定数量的包含文件名的表情图片列表
    """
    prefix, suffix = fileList[0].split('.')
    if prefix.count('_') > 0:
        length = len(fileList)
        if length > framenum:
            cut = length - framenum
            while cut > 0:
                fileList.pop()
                cut = cut - 1
                if cut == 0:
                    break
                del fileList[0]
                cut = cut - 1
            imgList = fileList
        else:
            imgBegin = onsetFrame
            imgFinish = offsetFrame
            while imgFinish - imgBegin + 1 < framenum:
                if apexFrame - onsetFrame - 1 > 0:
                    fileList.insert(apexFrame - onsetFrame, fileList[apexFrame - onsetFrame - 1])
                else:
                    fileList.insert(apexFrame - onsetFrame, fileList[apexFrame - onsetFrame])
                imgFinish = imgFinish + 1
            imgList = fileList[imgBegin - 1:imgFinish]
            print("imgList:" + str(len(imgList)))
            imgList = fileList
        print("imgList:" + str(len(imgList)))
    elif prefix.count('img') > 0:
        if type(apexFrame) is not int:
            apexFrame = int((onsetFrame+offsetFrame)/2)
        if offsetFrame - onsetFrame + 1 >= framenum:
            if apexFrame > int((framenum*1.0)/2):
                imgBegin = apexFrame - int((framenum*1.0/2))
                imgFinish = apexFrame + int((framenum*1.0/2))
                imgList = fileList[imgBegin - 1:imgFinish - 1]
                print("imgList:" + str(len(imgList)))
            else:
                imgBegin = 0
                imgFinish = apexFrame + int((framenum * 1.0 / 2))
                imgList = fileList[imgBegin:imgFinish - 1]
                print("imgList:" + str(len(imgList)))
        else:
            imgBegin = onsetFrame
            imgFinish = offsetFrame
            while imgFinish - imgBegin + 1 < framenum:
                fileList.insert(apexFrame, fileList[apexFrame - 1])
                imgFinish = imgFinish + 1
            imgList = fileList[imgBegin - 1:imgFinish]
            print("imgList:" + str(len(imgList)))
    else:
        if offsetFrame - onsetFrame + 1 > framenum:
            imgList = fileList[onsetFrame - firstFrame:offsetFrame - firstFrame]
            cut = len(imgList) - framenum
            while cut > 0:
                imgList.pop()
                cut = cut - 1
                if cut == 0:
                    break
                del imgList[0]
                cut = cut - 1
            print("imgList:" + str(len(imgList)))
        else:
            imgList = fileList[onsetFrame - firstFrame:offsetFrame - firstFrame]
            apexFrame = int((offsetFrame + onsetFrame) / 2 - onsetFrame)
            while len(imgList) < framenum:
                imgList.insert(apexFrame, imgList[apexFrame])
            print("imgList:" + str(len(imgList)))
        if onsetFrame2 is not None and offsetFrame2 is not None:
            if onsetFrame3 is not None and offsetFrame3 is not None:
                if offsetFrame2 - onsetFrame2 + 1 > framenum:
                    imgList2 = fileList[onsetFrame2 - onsetFrame:offsetFrame2 - onsetFrame]
                    cut = len(imgList2) - framenum
                    while cut > 0:
                        imgList2.pop()
                        cut = cut - 1
                        if cut == 0:
                            break
                        del imgList2[0]
                        cut = cut - 1
                else:
                    imgList2 = fileList[onsetFrame2 - onsetFrame:offsetFrame2 - onsetFrame]
                    apexFrame = int((offsetFrame2 + onsetFrame2) / 2 - onsetFrame2)
                    while len(imgList2) < framenum:
                        imgList2.insert(apexFrame, fileList[apexFrame])
                if offsetFrame3 - onsetFrame3 + 1 > framenum:
                    imgList3 = fileList[onsetFrame3 - onsetFrame2:offsetFrame3 - onsetFrame2]
                    cut = len(imgList3) - framenum
                    while cut > 0:
                        imgList3.pop()
                        cut = cut - 1
                        if cut == 0:
                            break
                        del imgList3[0]
                        cut = cut - 1
                    return imgList, imgList2, imgList3
                else:
                    imgList3 = fileList[onsetFrame3 - onsetFrame2:offsetFrame3 - onsetFrame2]
                    apexFrame = int((offsetFrame3 + onsetFrame3) / 2 - onsetFrame3)
                    while len(imgList3) < framenum:
                        imgList3.insert(apexFrame, imgList3[apexFrame])
                    print("imgList:" + str(len(imgList)) + ", imgList2:" + str(len(imgList2))+", imgList3:" + str(len(imgList3)))
                    return imgList, imgList2, imgList3
            else:
                if offsetFrame2 - onsetFrame2 + 1 > framenum:
                    imgList2 = fileList[onsetFrame2 - onsetFrame:offsetFrame2 - onsetFrame]
                    cut = len(imgList2) - framenum
                    while cut > 0:
                        imgList2.pop()
                        cut = cut - 1
                        if cut == 0:
                            break
                        del imgList2[0]
                        cut = cut - 1
                    return imgList, imgList2
                else:
                    imgList2 = fileList[onsetFrame2 - onsetFrame:offsetFrame2 - onsetFrame]
                    apexFrame = int((offsetFrame2 + onsetFrame2) / 2 - onsetFrame2)
                    print(len(imgList2))
                    while len(imgList2) < framenum:
                        imgList2.insert(apexFrame, imgList2[apexFrame - 1])
                    print("imgList:"+str(len(imgList))+", imgList2:"+str(len(imgList2)))
                    return imgList, imgList2
    return imgList
