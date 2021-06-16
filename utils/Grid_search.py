import numpy as np
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from datasets.casmeiiDatasets import load_CASMEII_data
from datasets.sammDatasets import load_SAMM_data
from datasets.smicDatasets import load_SMIC_data
from models.ShuffleNetV2 import ShuffleNetV2
from utils.dataPreparation import dataPraparation


def creat_model(optimizer='adam'):
    model = ShuffleNetV2(input_shape=(224, 224, 1), classes=3, scale_factor=1)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
    return model


mymodel = KerasClassifier(build_fn=creat_model, epochs=100, batch_size=30, verbose=0)
data = []
data.extend(load_CASMEII_data(LOO=True, framenum=30, OF=False))
data.extend(load_SAMM_data(LOO=True, framenum=30, OF=False))
data.extend(load_SMIC_data(LOO=True, framenum=30, OF=False))

trainFaceList, trainEmotionList, testFaceList, testEmotionList = dataPraparation(data, 1, 30, 30, LOO=True, OF=False)


# batch_size = [10, 40, 60, 80]
# epochs = [50, 80, 90, 100]

optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
#
# learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
# momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
#
# init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
#
# activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
#
# weight_constraint = [1, 2, 3, 4, 5]
# dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# param_grid_ = {"max_samples": np.arange(0.5, 1.0, 0.1), "n_estimators": np.arange(10, 200, 20), "bootstrap": ["true", "False"]}
param_grid_ = dict(optimizer=optimizer)
GS_bag = GridSearchCV(mymodel, param_grid_, cv=LeaveOneOut())
GS_bag.fit(np.asarray(trainFaceList), np.asarray(trainEmotionList))
# print(GS_bag.best_params_)
print("Best: %f using %s" % (GS_bag.best_score_, GS_bag.best_params_))
means = GS_bag.cv_results_['mean_test_score']
stds = GS_bag.cv_results_['std_test_score']
params = GS_bag.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
