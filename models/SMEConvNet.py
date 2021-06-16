import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers, models

from datasets.casmeiiDatasets import input_CASMEII_data


train_faces, train_emotions, test_faces, test_emotions, validation_faces, validation_emotions = input_CASMEII_data()

train_emotions = tf.one_hot(train_emotions, depth=4)
validation_emotions = tf.one_hot(validation_emotions, depth=4)
test_emotions = tf.one_hot(test_emotions, depth=4)

print("训练集图片数量：" + str(len(train_faces)) + "，标签数量：" + str(len(train_emotions)))
print("验证集图片数量：" + str(len(validation_faces)) + "，标签数量：" + str(len(validation_emotions)))
print("测试集图片数量：" + str(len(test_faces)) + "，标签数量：" + str(len(test_faces)))

model = models.Sequential()
model.add(layers.Conv2D(96, kernel_size=(11, 11), strides=2, activation='relu', input_shape=(224, 224, 1)))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(layers.Conv2D(96, kernel_size=(5, 5), strides=1, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(layers.Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(layers.Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(layers.Flatten())
model.add(layers.Dense(500, activation='relu'))
model.add(layers.Dense(500, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

model.summary()
# loss选择中：sparse_categorical_crossentropy要求标签为非one-hot编码，categorical_crossentropy要求标签为one-hot编码
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001, decay=0.005, momentum=0.9, nesterov=False),
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )

callbacks = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10,
                                                 verbose=0, mode='max', min_delta=0.001, cooldown=0, min_lr=0.000001)
result = model.fit(train_faces, train_emotions, batch_size=60,
                   epochs=30, callbacks=callbacks, validation_data=(validation_faces, validation_emotions))

plt.plot(result.history['accuracy'], label='accuracy')
plt.plot(result.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_faces, test_emotions, verbose=2)

print(test_acc)
