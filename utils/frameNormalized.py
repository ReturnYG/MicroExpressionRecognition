def frameNormalized(fileList, onsetFrame, offsetFrame, apexFrame=None, firstFrame=None, lastFrame=None,
                    onsetFrame2=None, offsetFrame2=None, onsetFrame3=None, offsetFrame3=None):
    prefix, suffix = fileList[0].split('.')
    if prefix.count('_') > 0:
        length = len(fileList)
        if length > 30:
            cut = length - 30
            while cut > 0:
                fileList.pop()
                cut = cut - 1
                if cut == 0:
                    break
                del fileList[0]
                cut = cut - 1
            imgList = fileList
        else:
            imgList = fileList
        print("imgList:" + str(len(imgList)))
    elif prefix.count('img') > 0:
        if type(apexFrame) is not int:
            apexFrame = int((onsetFrame+offsetFrame)/2)
        if offsetFrame - onsetFrame + 1 > 30:
            imgBegin = apexFrame - 15
            imgFinish = apexFrame + 15
            imgList = fileList[imgBegin - 1:imgFinish - 1]
            print("imgList:" + str(len(imgList)))
        else:
            imgBegin = onsetFrame
            imgFinish = offsetFrame
            while imgFinish - imgBegin + 1 < 30:
                fileList.insert(apexFrame, fileList[apexFrame - 1])
                imgFinish = imgFinish + 1
            imgList = fileList[imgBegin - 1:imgFinish - 1]
            print("imgList:" + str(len(imgList)))
    else:
        if offsetFrame - onsetFrame + 1 > 30:
            imgList = fileList[onsetFrame - firstFrame:offsetFrame - firstFrame]
            cut = len(imgList) - 30
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
            while len(imgList) < 30:
                imgList.insert(apexFrame, imgList[apexFrame])
            print("imgList:" + str(len(imgList)))
        if onsetFrame2 is not None and offsetFrame2 is not None:
            if onsetFrame3 is not None and offsetFrame3 is not None:
                if offsetFrame2 - onsetFrame2 + 1 > 30:
                    imgList2 = fileList[onsetFrame2 - onsetFrame:offsetFrame2 - onsetFrame]
                    cut = len(imgList2) - 30
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
                    while len(imgList2) < 30:
                        imgList2.insert(apexFrame, fileList[apexFrame])
                if offsetFrame3 - onsetFrame3 + 1 > 30:
                    imgList3 = fileList[onsetFrame3 - onsetFrame2:offsetFrame3 - onsetFrame2]
                    cut = len(imgList3) - 30
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
                    while len(imgList3) < 30:
                        imgList3.insert(apexFrame, imgList3[apexFrame])
                    print("imgList:" + str(len(imgList)) + ", imgList2:" + str(len(imgList2))+", imgList3:" + str(len(imgList3)))
                    return imgList, imgList2, imgList3
            else:
                if offsetFrame2 - onsetFrame2 + 1 > 30:
                    imgList2 = fileList[onsetFrame2 - onsetFrame:offsetFrame2 - onsetFrame]
                    cut = len(imgList2) - 30
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
                    while len(imgList2) < 30:
                        imgList2.insert(apexFrame, imgList2[apexFrame - 1])
                    print("imgList:"+str(len(imgList))+", imgList2:"+str(len(imgList2)))
                    return imgList, imgList2

    return imgList
