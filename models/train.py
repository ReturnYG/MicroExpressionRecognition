import itertools
import random
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from datasets.casmeiiDatasets import load_CASMEII_data
from datasets.sammDatasets import load_SAMM_data
from datasets.smicDatasets import load_SMIC_data
import tensorflow as tf
from transformers import DeiTModel, DeiTConfig
from models.MobileNetV3 import MobileNetV3_Small
from models.MobileVit_Keras import create_mobilevit

# 数据准备############################################################
data, faces, emos, innerfacelist, inneremolist = [], [], [], [], []
casme2 = load_CASMEII_data(LOO=False, framenum=31, OF=False, Apexonly=True)
samm = load_SAMM_data(LOO=False, framenum=31, OF=False, Apexonly=True)
smic = load_SMIC_data(LOO=False, framenum=31, OF=False, Apexonly=True)

innerfacelist.extend(casme2[0])
inneremolist.extend(casme2[1])
innerfacelist.extend(samm[0])
inneremolist.extend(samm[1])
innerfacelist.extend(smic[0])
inneremolist.extend(smic[1])

ziplist = list(zip(innerfacelist, inneremolist))
random.shuffle(ziplist)  # 打乱训练集
dataFacesTuple, dataEmotionsTuple = zip(*ziplist)

dataFacesList = list(dataFacesTuple)
dataEmotionsList = list(dataEmotionsTuple)

dataFacesList_exdim = []
dataEmotionsList_onhot = []
for i in dataFacesList:
    dataFacesList_exdim.append(np.expand_dims(i, -1))
for j in dataEmotionsList:
    dataEmotionsList_onhot.append(tf.one_hot(j, depth=3))

trainFaceList = dataFacesList_exdim[:380]
validFaceList = dataFacesList_exdim[380:]
trainEmoList = dataEmotionsList_onhot[:380]
validEmoList = dataEmotionsList_onhot[380:]

trainFaceArray = np.array(trainFaceList)
validFaceArray = np.array(validFaceList)
trainEmoArray = np.array(trainEmoList)
validEmoArray = np.array(validEmoList)

testFaceArray = np.array(dataFacesList_exdim)
testEmoArray = np.array(dataEmotionsList_onhot)
# 数据准备END#########################################################


# 绘制混淆矩阵
def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=plt.cm.Greens, normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(15, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    # 这里这个savefig是保存图片，如果想把图存在什么地方就改一下下面的路径，然后dpi设一下分辨率即可。
    plt.savefig('/Users/returnyg/PycharmProjects/MicroExpressionRecognition/models/Confusion Matrix Image/result/confusion_matrix.png', dpi=350)
    plt.show()


# 定义学习率函数
def scheduler(epochs, lr):
    if epochs < 50:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


# 显示混淆矩阵
def plot_confuse(model, x_val, y_val, labels, normalize=True):
    y_ = model.predict(x_val, batch_size=16)
    predictions = np.argmax(y_, axis=1)
    truelabel = np.argmax(y_val, axis=1)
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
    plt.figure()
    plot_confusion_matrix(conf_mat, normalize=normalize, target_names=labels, title='Confusion Matrix')


# 部分参数定义
EPOCHS = 100
checkpoint_filepath = '/Users/returnyg/PycharmProjects/MicroExpressionRecognition/models/checkpoint'
labels = ["Positive", "Negative", "Surprise"]


# mobilevit训练函数
def train_model():
    model = create_mobilevit(num_classes=3)
    # model = tf.keras.models.load_model("/Users/returnyg/PycharmProjects/MicroExpressionRecognition/models/checkpoint")
    model.summary()
    my_callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_acc', save_weights_only=False, mode='auto'),
                    tf.keras.callbacks.LearningRateScheduler(scheduler),
                    tf.keras.callbacks.TerminateOnNaN(),
                    tf.keras.callbacks.TensorBoard(log_dir="/Users/returnyg/PycharmProjects/MicroExpressionRecognition/models/log", histogram_freq=1, update_freq='batch'),
                    ]

    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01), loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['acc'])
    model.fit(trainFaceArray, trainEmoArray, batch_size=16, epochs=EPOCHS, shuffle=True, validation_data=(validFaceArray, validEmoArray), callbacks=my_callbacks)
    plot_confuse(model, testFaceArray, testEmoArray, labels)


# mobilenetv3训练函数
def train_mobilenetv3():
    model = MobileNetV3_Small(shape=(256, 256, 1), n_class=3, alpha=1.0, include_top=True)
    model = model.build()
    model.summary()
    my_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_acc', save_weights_only=False, mode='auto'),
        tf.keras.callbacks.LearningRateScheduler(scheduler),
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.TensorBoard(log_dir="/Users/returnyg/PycharmProjects/MicroExpressionRecognition/models/log", histogram_freq=1, update_freq='batch'),
        ]

    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['acc'])
    model.fit(trainFaceArray, trainEmoArray, batch_size=16, epochs=EPOCHS, shuffle=True, validation_data=(validFaceArray, validEmoArray), callbacks=my_callbacks)
    plot_confuse(model, testFaceArray, testEmoArray, labels)


# Deit训练函数
def train_Deit():
    configuration = DeiTConfig()
    model = DeiTModel(configuration)
    configuration = model.config


train_model()
# train_mobilenetv3()
# model = tf.keras.models.load_model("/Users/returnyg/PycharmProjects/MicroExpressionRecognition/models/checkpoint")
# plot_confuse(model, testFaceArray, testEmoArray, labels, normalize=True)
# model.summary()
