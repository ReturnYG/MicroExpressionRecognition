import datetime
import itertools
import os
import time
import numpy as np
from tensorflow.python.keras.applications.vgg16 import VGG16

from datasets.casmeiiDatasets import load_CASMEII_data
from datasets.sammDatasets import load_SAMM_data
from datasets.smicDatasets import load_SMIC_data
from models.ShuffleNetV2 import ShuffleNetV2
import tensorflow as tf
import matplotlib.pyplot as plt
from numpy import random
# from models.shuffleNet_v2_opconty import ShuffleNetV2
from utils.batchSizeFix import batchFuc
from sklearn.metrics import confusion_matrix

# 读取数据集
from utils.build_result import generate_result
from utils.dataPreparation import dataPraparation, flatten
from utils.sampleBalance import sample_balance

data = []
data.extend(load_CASMEII_data(LOO=True, framenum=30, OF=False))
data.extend(load_SAMM_data(LOO=True, framenum=30, OF=False))
data.extend(load_SMIC_data(LOO=True, framenum=30, OF=False))
# 定义分类的个数，用于混淆矩阵
class_name = [0, 1, 2]


# 动态调整学习率函数
def dynamic_learning_rate(epoch, now_learning_rate, training_accuracy, max_accuracy):
    min_learning_rate = 0.0001  # 最低学习率
    max_learning_rate = 0.05  # 最高学习率
    # 前15个epoch不进行调整
    if epoch < 6 and training_accuracy < 0.5:
        new_learning_rate = now_learning_rate
        return new_learning_rate
    # 根据准确率进行调整
    elif 11 < epoch and training_accuracy > 0.5:
        if training_accuracy < 0.6:
            new_learning_rate = now_learning_rate * 0.99
            if new_learning_rate > max_learning_rate:
                return max_learning_rate
            else:
                return new_learning_rate
        if training_accuracy > max_accuracy and training_accuracy > 0.6:
            max_accuracy = training_accuracy
            new_learning_rate = now_learning_rate * 0.9
            if new_learning_rate > min_learning_rate:
                return new_learning_rate
            else:
                return min_learning_rate
        elif training_accuracy < max_accuracy:
            new_learning_rate = now_learning_rate * 1.0
            return new_learning_rate
    else:
        new_learning_rate = now_learning_rate * 0.99
        return new_learning_rate


# 保存并显示混淆矩阵
def plot_confusion_matrix(cm, now_time, epoch, target_names, status, title='Confusion matrix', cmap=plt.cm.Blues,
                          normalize=True):
    class_label = ['Positive', 'Negative', 'Surprise']
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
        plt.xticks(tick_marks, class_label, rotation=45)
        plt.yticks(tick_marks, class_label)

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
    image_name = "/Users/returnyg/PycharmProjects/MicroExpressionRecognition/models/Confusion Matrix Image/" + status + "/cm" + now_time + "第" + str(epoch) + "个epoch.png"
    plt.savefig(image_name, format='png')
    plt.close()
    file_contents = tf.io.read_file(image_name)
    image = tf.image.decode_png(file_contents, channels=4)
    image = tf.expand_dims(image, 0)
    return image


# 训练循环
def trainMyModel_Custom(dataset, datasetName, learning_rate, framenum=30):
    newModel = ShuffleNetV2(input_shape=(224, 224, 1), scale_factor=1, classes=3)  # 读取模型
    newModel.summary()
    # newModel = VGG16(input_shape=(224, 224, 1), include_top=True, weights=None, classes=3)
    # 读取前序训练的权重
    if os.listdir('/Users/returnyg/PycharmProjects/MicroExpressionRecognition/models/save_weight').count('max_acc_' + datasetName + '.h5') > 0:
        print("加载上次准确率最高的权重")
        newModel.load_weights('/Users/returnyg/PycharmProjects/MicroExpressionRecognition/models/save_weight/max_acc_' + datasetName + '.h5')
        print("权重加载成功")
    max_accuracy = 0  # 初始化最高准确率
    print("当前在{}数据集进行训练。".format(datasetName))
    loss_fuc = tf.keras.losses.CategoricalCrossentropy(from_logits=True)  # 设定损失函数
    val_acc_metric = tf.keras.metrics.CategoricalAccuracy()  # 设定验证准确率计算函数
    epoch_loss_avg = tf.keras.metrics.Mean()  # 设定epoch平均损失函数
    epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()  # 设定epoch准确率函数
    # 设定优化器
    # optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=False, name='SGD')
    # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-04, amsgrad=True, name='Adam')
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9, momentum=0.0, epsilon=1e-7, centered=True, name="RMSprop")
    # optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate, rho=0.9, epsilon=1e-7, clipnorm=1., name="Adadelta")
    # 初始化TensorBoard相关参数
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = '/Users/returnyg/PycharmProjects/MicroExpressionRecognition/models/log11.1/train' + current_time
    test_log_dir = '/Users/returnyg/PycharmProjects/MicroExpressionRecognition/models/log11.1/test' + current_time
    cm_train_log_dir = '/Users/returnyg/PycharmProjects/MicroExpressionRecognition/models/log11.1/cm/train' + current_time
    cm_val_log_dir = '/Users/returnyg/PycharmProjects/MicroExpressionRecognition/models/log11.1/cm/val' + current_time
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    cm_train_summary_writer = tf.summary.create_file_writer(cm_train_log_dir)
    cm_val_summary_writer = tf.summary.create_file_writer(cm_val_log_dir)

    # 计算损失
    def loss(model, x, y):
        y_ = model(x)
        # y_pred = tf.clip_by_value(y_, 10e-8, 1.-10e-8)
        y_new = generate_result(y_, y)
        return loss_fuc(y_true=y, y_pred=y_), y_new

    # 计算梯度
    def grad(model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value, y_pred = loss(model, inputs, targets)
        return loss_value, tape.gradient(loss_value, model.trainable_variables), y_pred

    train_loss_result = []
    train_accuracy_result = []
    max_acc = 0  # 初始化最高准确率
    # 训练迭代循环
    for epoch in range(0, 500):  # len(dataset)
        print("现在是第{}个epoch，共有{}个epoch。".format(epoch + 1, 500))
        trainFaceList, trainEmotionList, testFaceList, testEmotionList = dataPraparation(dataset, epoch, framenum, 30, LOO=True, OF=False)

        step = 1  # 训练阶段
        train_y_true = []
        train_y_pred = []
        # 训练循环
        for x, y in zip(trainFaceList, trainEmotionList):
            loss_value, grads, y_pred = grad(newModel, x, y)  # 计算损失和梯度
            train_y_true.append(np.argmax(y, axis=1).tolist())
            train_y_pred.append(np.argmax(y_pred, axis=1).tolist())
            epoch_loss_avg.update_state(loss_value)  # 记录损失
            epoch_accuracy.update_state(y, y_pred)  # 记录准确率
            optimizer.apply_gradients(zip(grads, newModel.trainable_variables))  # 优化器更新权重
            end_time = time.time()

            if step % 100 == 0:
                print("当前是第{}个batch，还有{}个batch待做".format(step, len(trainFaceList) - step))
            # Tensorboard显示损失和准确率
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', epoch_loss_avg.result(), step=epoch)
                tf.summary.scalar('accuracy', epoch_accuracy.result(), step=epoch)
            step = step + 1
        # 每个epoch后记录相关数据
        train_loss_result.append(epoch_loss_avg.result())
        train_accuracy_result.append(epoch_accuracy.result())
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch + 1, epoch_loss_avg.result(), epoch_accuracy.result()))

        train_y_true = flatten(train_y_true)
        train_y_pred = flatten(train_y_pred)
        cm = confusion_matrix(y_true=train_y_true, y_pred=train_y_pred, labels=class_name)  # 获取混淆矩阵
        now_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        # 生成PNG文件并显示在TensorBoard
        train_cm_image = plot_confusion_matrix(cm, now_time=now_time, epoch=epoch + 1, status="Train", normalize=False,
                                               target_names=class_name, title="Confusion Matrix")
        with cm_train_summary_writer.as_default():
            tf.summary.image("Train Confusion Matrix", train_cm_image, step=epoch + 1)
        # 根据准确率判是否保存权重
        if epoch_accuracy.result() > max_acc:
            print(f"第{epoch + 1}个epoch的训练准确率为：{epoch_accuracy.result():.3%}，比本次训练历史最高准确率{max_acc:.3%}高，保存权重。")
            max_acc = epoch_accuracy.result()
            # 保存准确率较高的权重
            newModel.save_weights('/Users/returnyg/PycharmProjects/MicroExpressionRecognition/models/save_weight/max_acc_' + datasetName + '.h5', overwrite=True, save_format='h5')

        epoch_accuracy.reset_states()
        epoch_loss_avg.reset_states()

        val_y_true = []
        val_y_pred = []
        # 测试循环
        for x_val, y_val in zip(testFaceList, testEmotionList):
            val_logits = newModel(x_val, training=False)
            val_acc_metric.update_state(y_val, val_logits)
            val_y_true.append(np.argmax(y_val, axis=1).tolist())
            val_y_pred.append(np.argmax(val_logits, axis=1).tolist())

        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()

        val_y_true = flatten(val_y_true)
        val_y_pred = flatten(val_y_pred)
        cm = confusion_matrix(y_true=val_y_true, y_pred=val_y_pred, labels=class_name)
        now_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        plt.figure()
        val_cm_image = plot_confusion_matrix(cm, now_time=now_time, epoch=epoch + 1, status="Test", normalize=False, target_names=class_name, title="Confusion Matrix")
        with cm_val_summary_writer.as_default():
            tf.summary.image("Val Confusion Matrix", val_cm_image, step=epoch + 1)

        with test_summary_writer.as_default():
            tf.summary.scalar('accuracy', val_acc, step=epoch + 1)

        print("Validation acc: %.4f" % (float(val_acc),))
        # 根据准确率判学习率的变化
        learning_rate = dynamic_learning_rate(epoch, learning_rate, training_accuracy=float(epoch_accuracy.result()), max_accuracy=max_accuracy)
        if float(epoch_accuracy.result()) < 0.4:
            newModel.save("/Users/returnyg/PycharmProjects/MicroExpressionRecognition/models/error_model")
            # sys.exit("Bad Model! Please Check Weight and Loss!")
        print(f"运行完第{epoch + 1}个epoch后，学习率更新为{learning_rate:.4f}")
    return train_accuracy_result, train_loss_result


def trainMyModel_OF(dataset, datasetName, learning_rate, framenum=30):
    newModel = ShuffleNetV2(input_shape=(224, 224, 1), scale_factor=1, classes=3)  # 读取模型
    # 读取前序训练的权重
    if os.listdir('/Users/returnyg/PycharmProjects/MicroExpressionRecognition/models/save_weight').count('max_acc_' + datasetName + '.h5') > 0:
        print("加载上次准确率最高的权重")
        newModel.load_weights('/Users/returnyg/PycharmProjects/MicroExpressionRecognition/models/save_weight/max_acc_' + datasetName + '.h5')
        print("权重加载成功")
    max_accuracy = 0  # 初始化最高准确率
    print("当前在{}数据集进行训练。".format(datasetName))
    loss_fuc = tf.keras.losses.CategoricalCrossentropy(from_logits=True)  # 设定损失函数
    val_acc_metric = tf.keras.metrics.CategoricalAccuracy()  # 设定验证准确率计算函数
    epoch_loss_avg = tf.keras.metrics.Mean()  # 设定epoch平均损失函数
    epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()  # 设定epoch准确率函数
    # 设定优化器
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=False, name='SGD')
    # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-04, amsgrad=True, name='Adam')
    # optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9, momentum=0.0, epsilon=1e-7, clipnorm=1., centered=True, name="RMSprop")
    # optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate, rho=0.9, epsilon=1e-7, clipnorm=1., name="Adadelta")
    # 初始化TensorBoard相关参数
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = '/Users/returnyg/PycharmProjects/MicroExpressionRecognition/models/log/train' + current_time
    test_log_dir = '/Users/returnyg/PycharmProjects/MicroExpressionRecognition/models/log/test' + current_time
    cm_train_log_dir = '/Users/returnyg/PycharmProjects/MicroExpressionRecognition/models/log/cm/train' + current_time
    cm_val_log_dir = '/Users/returnyg/PycharmProjects/MicroExpressionRecognition/models/log/cm/val' + current_time
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    cm_train_summary_writer = tf.summary.create_file_writer(cm_train_log_dir)
    cm_val_summary_writer = tf.summary.create_file_writer(cm_val_log_dir)

    # 计算损失
    def loss(model, x, y):
        y_ = model(x)
        # y_pred = tf.clip_by_value(y_, 10e-8, 1.-10e-8)
        return loss_fuc(y_true=y, y_pred=y_), y_

    # 计算梯度
    def grad(model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value, y_pred = loss(model, inputs, targets)
        return loss_value, tape.gradient(loss_value, model.trainable_variables), y_pred

    train_loss_result = []
    train_accuracy_result = []
    max_acc = 0  # 初始化最高准确率
    # 训练迭代循环
    for epoch in range(0, 500):  # len(dataset)
        print("现在是第{}个epoch，共有{}个epoch。".format(epoch + 1, 500))
        trainFaceList, trainEmotionList, testFaceList, testEmotionList = dataPraparation(dataset, epoch, framenum, 30,
                                                                                         LOO=False, OF=True)

        step = 1  # 训练阶段
        train_y_true = []
        train_y_pred = []
        # 训练循环
        for x, y in zip(trainFaceList, trainEmotionList):
            loss_value, grads, y_pred = grad(newModel, x, y)  # 计算损失和梯度
            train_y_true.append(np.argmax(y, axis=1).tolist())
            train_y_pred.append(np.argmax(y_pred, axis=1).tolist())
            epoch_loss_avg.update_state(loss_value)  # 记录损失
            epoch_accuracy.update_state(y, y_pred)  # 记录准确率
            optimizer.apply_gradients(zip(grads, newModel.trainable_variables))  # 优化器更新权重

            if step % 100 == 0:
                print("当前是第{}个batch，还有{}个batch待做".format(step, len(trainFaceList) - step))
            # Tensorboard显示损失和准确率
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', epoch_loss_avg.result(), step=epoch)
                tf.summary.scalar('accuracy', epoch_accuracy.result(), step=epoch)
            step = step + 1
        # 每个epoch后记录相关数据
        train_loss_result.append(epoch_loss_avg.result())
        train_accuracy_result.append(epoch_accuracy.result())
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch + 1, epoch_loss_avg.result(), epoch_accuracy.result()))

        train_y_true = flatten(train_y_true)
        train_y_pred = flatten(train_y_pred)
        cm = confusion_matrix(y_true=train_y_true, y_pred=train_y_pred, labels=class_name)  # 获取混淆矩阵
        now_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        # 生成PNG文件并显示在TensorBoard
        train_cm_image = plot_confusion_matrix(cm, now_time=now_time, epoch=epoch + 1, status="Train", normalize=False,
                                               target_names=class_name, title="Confusion Matrix")
        with cm_train_summary_writer.as_default():
            tf.summary.image("Train Confusion Matrix", train_cm_image, step=epoch + 1)
        # 根据准确率判是否保存权重
        if epoch_accuracy.result() > max_acc:
            print(f"第{epoch + 1}个epoch的训练准确率为：{epoch_accuracy.result():.3%}，比本次训练历史最高准确率{max_acc:.3%}高，保存权重。")
            max_acc = epoch_accuracy.result()
            # 保存准确率较高的权重
            newModel.save_weights('/Users/returnyg/PycharmProjects/MicroExpressionRecognition/models/save_weight/max_acc_' + datasetName + '.h5', overwrite=True, save_format='h5')

        epoch_accuracy.reset_states()
        epoch_loss_avg.reset_states()

        val_y_true = []
        val_y_pred = []
        # 测试循环
        for x_val, y_val in zip(testFaceList, testEmotionList):
            val_logits = newModel(x_val, training=False)
            val_acc_metric.update_state(y_val, val_logits)
            val_y_true.append(np.argmax(y_val, axis=1).tolist())
            val_y_pred.append(np.argmax(val_logits, axis=1).tolist())

        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()

        val_y_true = flatten(val_y_true)
        val_y_pred = flatten(val_y_pred)
        cm = confusion_matrix(y_true=val_y_true, y_pred=val_y_pred, labels=class_name)
        now_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        plt.figure()
        val_cm_image = plot_confusion_matrix(cm, now_time=now_time, epoch=epoch + 1, status="Test", normalize=False, target_names=class_name, title="Confusion Matrix")
        with cm_val_summary_writer.as_default():
            tf.summary.image("Val Confusion Matrix", val_cm_image, step=epoch + 1)

        with test_summary_writer.as_default():
            tf.summary.scalar('accuracy', val_acc, step=epoch + 1)

        print("Validation acc: %.4f" % (float(val_acc),))
        # 根据准确率判学习率的变化
        learning_rate = dynamic_learning_rate(epoch, learning_rate, training_accuracy=float(epoch_accuracy.result()), max_accuracy=max_accuracy)
        if float(epoch_accuracy.result()) < 0.4:
            newModel.save("/Users/returnyg/PycharmProjects/MicroExpressionRecognition/models/error_model")
            # sys.exit("Bad Model! Please Check Weight and Loss!")
        print(f"运行完第{epoch + 1}个epoch后，学习率更新为{learning_rate:.4f}")
    return train_accuracy_result, train_loss_result


def trainMyModel_Auto(dataset, dataName, learning_rate=0.001, framnum=30):
    trainset, testset = [], []
    # newModel = ShuffleNetV2(scale_factor=1, classes=3)  # 读取模型
    newModel = VGG16(input_shape=(224, 224, 3), include_top=True, weights=None, classes=3)
    testset.extend(dataset[0:10])
    trainset = list(dataset)
    for i in range(0, 11):
        trainset.pop(i)

    def flatten(input_list):
        output_list = []
        while True:
            if not input_list:
                break
            for index, i in enumerate(input_list):
                if type(i) == list:
                    input_list = i + input_list[index + 1:]
                    break
                else:
                    output_list.append(i)
                    input_list.pop(index)
                    break
        return output_list

    testfaces = []
    testemotions = []
    trainfaces = []
    trainemotions = []
    for subject in testset:
        testfaces.append(subject[1])
        testemotions.append(subject[2])
    for subject in trainset:
        trainfaces.append(subject[1])
        trainemotions.append(subject[2])
    trainfaceslist = flatten(trainfaces)
    trainemotionslist = flatten(trainemotions)
    testfaceslist = flatten(testfaces)
    testemotionslist = flatten(testemotions)

    trainemotionslist_onehot, testemotionslist_onehot = [], []
    for i in trainemotionslist:
        trainemotionslist_onehot.append(tf.one_hot(i, depth=3))
    for i in testemotionslist:
        testemotionslist_onehot.append(tf.one_hot(i, depth=3))

    trainfaceslist = np.asarray(trainfaceslist)
    trainemotionslist = np.asarray(trainemotionslist_onehot)
    testfaceslist = np.asarray(testfaceslist)
    testemotionslist = np.asarray(testemotionslist_onehot)

    # batch_size_twice = 30
    # trainemotionslist_withBatch = batchFuc(trainemotionslist, batch_size_twice)
    # testemotionslist_withBatch = batchFuc(testemotionslist, batch_size_twice)
    # trainfaceslist_withBatch = batchFuc(trainfaceslist, batch_size_twice)
    # testfaceslist_withBatch = batchFuc(testfaceslist, batch_size_twice)

    # ziplist = list(zip(trainfaceslist_withBatch, trainemotionslist_withBatch))
    # random.shuffle(ziplist)  # 打乱训练集
    # trainfaceslist_withBatch_shuffle, trainemotionslist_withBatch_shuffle = zip(*ziplist)
    # ziplist = list(zip(testfaceslist_withBatch, testemotionslist_withBatch))
    # random.shuffle(ziplist)  # 打乱训练集
    # testfaceslist_withBatch_shuffle, testemotionslist_withBatch_shuffle = zip(*ziplist)

    newModel.compile(optimizer=tf.keras.optimizers.SGD(lr=0.005),
                     loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                     metrics=['acc'])
    result = newModel.fit(trainfaceslist, trainemotionslist, 30, 100, validation_data=(testfaceslist, testemotionslist))


train_accuracy_result, train_loss_result = trainMyModel_Custom(data, "Combine", learning_rate=0.001, framenum=30)
# trainMyModel_Auto(data, "Combine")
# train_accuracy_result, train_loss_result = trainMyModel_OF(data, "OF", learning_rate=0.0015, framenum=2)
