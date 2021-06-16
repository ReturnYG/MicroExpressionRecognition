# 调整batch_size
def batchFuc(fixlist, batch_size):
    afterBatchList = []
    group_num = len(fixlist) // batch_size
    for i in range(0, group_num*batch_size, batch_size):
        innerlist = fixlist[i:i+batch_size]
        afterBatchList.append(innerlist)
    if batch_size * group_num == len(fixlist):
        return afterBatchList
    else:
        afterBatchList.append(fixlist[batch_size * group_num:len(fixlist)])
        return afterBatchList
