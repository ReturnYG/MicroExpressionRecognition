
# 平衡情绪样本数量
def sample_balance(trainFaceList, trainEmoList):
    newTrainFaceList, newTrainEmoList = [], []
    emo_num = [0, 0, 0]
    for i in range(len(trainFaceList)):
        if trainEmoList[i][0] == 0:
            emo_num[0] += 1
        elif trainEmoList[i][0] == 1:
            emo_num[1] += 1
        else:
            emo_num[2] += 1
    print("平衡前共有{}个positive表情，{}个negative表情，{}个surprise表情".format(emo_num[0], emo_num[1], emo_num[2]))
    min_class = min(emo_num)  # 1
    new_pos, new_neg, new_sup = 0, 0, 0
    for j in range(len(trainFaceList)):
        if trainEmoList[j][0] == 0 and new_pos < min_class:
            newTrainFaceList.append(trainFaceList[j])
            newTrainEmoList.append(trainEmoList[j])
            new_pos += 1
        elif trainEmoList[j][0] == 1 and new_neg < min_class:
            newTrainFaceList.append(trainFaceList[j])
            newTrainEmoList.append(trainEmoList[j])
            new_neg += 1
        elif trainEmoList[j][0] == 2 and new_sup < min_class:
            newTrainFaceList.append(trainFaceList[j])
            newTrainEmoList.append(trainEmoList[j])
            new_sup += 1
    print("平衡后共有{}个positive表情，{}个negative表情，{}个surprise表情".format(new_pos, new_neg, new_sup))
    return newTrainFaceList, newTrainEmoList

