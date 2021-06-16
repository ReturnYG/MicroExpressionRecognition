import tensorflow as tf
from matplotlib import pyplot as plt

from datasets.casmeiiDatasets import input_CASMEII_data
# from datasets.sammDatasets import input_SAMM_data
# from datasets.smicDatasets import input_SMIC_data
from models.googlenet_keras import InceptionV3
from models.resnet_keras import ResNet50V2
from models.vgg16_keras import VGG16

train_faces, train_emotions, test_faces, test_emotions, validation_faces, validation_emotions = input_CASMEII_data()
# train_faces, train_emotions, test_faces, test_emotions, validation_faces, validation_emotions = input_SAMM_data()
# train_faces, train_emotions, test_faces, test_emotions, validation_faces, validation_emotions = input_SMIC_data()

train_emotions = tf.one_hot(train_emotions, depth=4)
validation_emotions = tf.one_hot(validation_emotions, depth=4)
test_emotions = tf.one_hot(test_emotions, depth=4)

print("训练集图片数量：" + str(len(train_faces)) + "，标签数量：" + str(len(train_emotions)))
print("验证集图片数量：" + str(len(validation_faces)) + "，标签数量：" + str(len(validation_emotions)))
print("测试集图片数量：" + str(len(test_faces)) + "，标签数量：" + str(len(test_faces)))


def vgg16_train(train_faces, train_emotions, test_faces, test_emotions, validation_faces, validation_emotions):
    vgg = VGG16(include_top=True,
                weights=None,
                input_tensor=None,
                input_shape=(224, 224, 1),
                pooling=None,
                classes=4,
                classifier_activation='softmax')
    vgg.summary()
    vgg.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00001,
                                                      amsgrad=False),
                   loss='categorical_crossentropy',
                   metrics=['categorical_accuracy'])
    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.1, patience=10,
                                                      verbose=1, mode='max', min_delta=0.001, cooldown=0,
                                                      min_lr=0.000001),
                 tf.keras.callbacks.ModelCheckpoint('checkpoint', monitor='val_categorical_accuracy', verbose=1,
                                                    save_best_only=True,
                                                    save_weights_only=True, mode='auto', save_freq='epoch',
                                                    options=None)]
    result = vgg.fit(train_faces, train_emotions, batch_size=30, epochs=30, callbacks=callbacks,
                        validation_data=(validation_faces, validation_emotions))
    show_result(result)
    # vgg.load_weights('checkpoint')
    test_loss, test_acc = vgg.evaluate(test_faces, test_emotions)
    print("测试准确率为" + str(test_acc))
    print("测试损失为" + str(test_loss))


def ResNet_train(train_faces, train_emotions, test_faces, test_emotions, validation_faces, validation_emotions):
    resnet = ResNet50V2(include_top=True,
                        weights=None,
                        input_tensor=None,
                        input_shape=(224, 224, 1),
                        pooling=None,
                        classes=4,
                        classifier_activation='softmax')
    resnet.summary()
    resnet.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00001,
                                                      amsgrad=False),
                   loss='categorical_crossentropy',
                   metrics=['categorical_accuracy'])
    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.1, patience=10,
                                                      verbose=1, mode='max', min_delta=0.001, cooldown=0,
                                                      min_lr=0.000001),
                 tf.keras.callbacks.ModelCheckpoint('checkpoint', monitor='val_categorical_accuracy', verbose=1,
                                                    save_best_only=True,
                                                    save_weights_only=False, mode='auto', save_freq='epoch',
                                                    options=None)]
    result = resnet.fit(train_faces, train_emotions, batch_size=30, epochs=30, callbacks=callbacks,
                        validation_data=(validation_faces, validation_emotions))
    show_result(result)
    # resnet.load_weights('checkpoint')
    test_loss, test_acc = resnet.evaluate(test_faces, test_emotions)
    print("测试准确率为" + str(test_acc))
    print("测试损失为" + str(test_loss))


def googleNet_train(train_faces, train_emotions, test_faces, test_emotions, validation_faces, validation_emotions):
    google_net = InceptionV3(include_top=True,
                             weights=None,
                             input_tensor=None,
                             input_shape=(224, 224, 1),
                             pooling=None,
                             classes=4,
                             classifier_activation='softmax')

    google_net.summary()
    google_net.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.000001,
                                           amsgrad=False),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )

    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.1, patience=10,
                                                      verbose=1, mode='max', min_delta=0.001, cooldown=0,
                                                      min_lr=0.000001),
                 tf.keras.callbacks.ModelCheckpoint('checkpoint', monitor='val_categorical_accuracy', verbose=1,
                                                    save_best_only=True,
                                                    save_weights_only=False, mode='auto', save_freq='epoch',
                                                    options=None)]
    result = google_net.fit(train_faces, train_emotions, batch_size=60,
                            epochs=30, callbacks=callbacks, validation_data=(validation_faces, validation_emotions))
    show_result(result)
    # google_net.load_weights('checkpoint')
    test_loss, test_acc = google_net.evaluate(test_faces, test_emotions, verbose=1)
    print("测试准确率为" + str(test_acc))
    print("测试损失为" + str(test_loss))


def show_result(result):
    plt.plot(result.history['categorical_accuracy'], label='categorical_accuracy')
    plt.plot(result.history['val_categorical_accuracy'], label='val_categorical_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()


googleNet_train(train_faces, train_emotions, test_faces, test_emotions, validation_faces, validation_emotions)
# ResNet_train(train_faces, train_emotions, test_faces, test_emotions, validation_faces, validation_emotions)
# vgg16_train(train_faces, train_emotions, test_faces, test_emotions, validation_faces, validation_emotions)
