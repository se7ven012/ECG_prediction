#%%
import h5py
import matplotlib
import sklearn
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model

# for cuDNN debug
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# for cuDNN debug

print("Versions of key libraries")
print("---")
print("tensorflow: ", tf.__version__)
print("numpy:      ", np.__version__)
print("matplotlib: ", matplotlib.__version__)
print("sklearn:    ", sklearn.__version__)

# plt setup
plt.style.use("seaborn")
plt.rcParams["ytick.right"] = True
plt.rcParams["ytick.labelright"] = True
plt.rcParams["ytick.left"] = False
plt.rcParams["ytick.labelleft"] = False
# Set the figure size to be 7 inch for (width,height)
plt.rcParams["figure.figsize"] = [7, 7]

print("Matplotlib setup completes.")
#%%
# load data
f = h5py.File("C:/Users/Administrator/Desktop/ECG_Prediction/cad5sec.mat", "r")
X = f["data"]
Y = f["classLabel"]
data = np.array(X)
label = np.array(Y)
data = np.transpose(data)
label = np.transpose(label)
nor = data[0:32000]
cad = data[32000:38120]
print("The shape of nor is", nor.shape, "and the data type is", nor.dtype)
print("The shape of cad is", cad.shape, "and the data type is", cad.dtype)

# 制作时序数据
def makeSteps(dat, length, dist):
    width = dat.shape[1]
    numOfSteps = int(np.floor((width - length) / dist) + 1)
    segments = np.zeros([dat.shape[0], numOfSteps, length], dtype=dat.dtype)
    for l in range(numOfSteps):
        segments[:, l, :] = dat[:, (l * dist) : (l * dist + length)]
    return segments


# Data pre-processing
trNor = nor[0:28800].copy()
tsNor = nor[28800:32000].copy()
trCad = cad[0:5000].copy()
tsCad = cad[5000:6120].copy()

length = 24
dist = 6
trNorS = makeSteps(trNor, length, dist)
tsNorS = makeSteps(tsNor, length, dist)
trCadS = makeSteps(trCad, length, dist)
tsCadS = makeSteps(tsCad, length, dist)

trDat = np.vstack([trNorS, trCadS])
tsDat = np.vstack([tsNorS, tsCadS])

trLbl = np.vstack([np.zeros([trNorS.shape[0], 1]), np.ones([trCadS.shape[0], 1])])
tsLbl = np.vstack([np.zeros([tsNorS.shape[0], 1]), np.ones([tsCadS.shape[0], 1])])

print("The shape of trDat is", trDat.shape, "and the type is", trDat.dtype)
print("The shape of trLbl is", trLbl.shape, "and the type is", trLbl.dtype)
print("")
print("The shape of tsDat is", tsDat.shape, "and the type is", tsDat.dtype)
print("The shape of tsLbl is", tsLbl.shape, "and the type is", tsLbl.dtype)

#%%
# Define model
modelname = "CNN_LSTM_V1"


def createModel():
    inputs = Input(shape=(trDat.shape[1], length))
    y = Conv1D(32, 5, activation="relu")(inputs)
    y = Dropout(0.25)(y)
    y = Conv1D(32, 5, activation="relu")(y)
    y = MaxPooling1D(2)(y)
    y = Conv1D(48, 5, activation="relu")(y)
    y = Dropout(0.5)(y)
    y = Conv1D(48, 5, activation="relu")(y)
    y = MaxPooling1D(2)(y)
    y = Conv1D(64, 5, activation="relu")(y)
    y = Dropout(0.5)(y)
    y = Conv1D(64, 5, activation="relu")(y)
    y = MaxPooling1D(2)(y)
    y = LSTM(8, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)(y)
    y = LSTM(4, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)(y)
    y = LSTM(2)(y)
    y = Dense(1, activation="sigmoid")(y)
    model = Model(inputs=inputs, outputs=y)

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


model = createModel()
model.summary()

#%%
# Define callback
folderpath = "C:/Users/Administrator/Desktop/ECG_Prediction/"
filepath = folderpath + modelname + ".hdf5"
checkpoint = ModelCheckpoint(
    filepath, monitor="val_accuracy", verbose=0, save_best_only=False, mode="max"
)

csv_logger = CSVLogger(folderpath + modelname + ".csv")

from keras.callbacks import ReduceLROnPlateau  # ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.1, patience=5, mode="auto", min_lr=0.00001
)

callbacks_list = [checkpoint, csv_logger, reduce_lr]

print("Callbacks created:")
print(callbacks_list[0])
print(callbacks_list[1])
print("")
print("Path to model:", filepath)
print("Path to log:  ", folderpath + modelname + ".csv")

#%%
model.fit(
    trDat,
    trLbl,
    validation_data=(tsDat, tsLbl),
    epochs=40,
    batch_size=128,
    shuffle=True,
    callbacks=callbacks_list,
)

#%%
model.load_weights(filepath)
print("Model weights loaded from:", filepath)

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

predicts = model.predict(tsDat)
print("Prediction completes.")

# print classification metrics
labelname = ["Normal", "CAD"]

testScores = metrics.accuracy_score(tsLbl, predicts.round())

print("Best accuracy (on testing dataset): %.2f%%" % (testScores * 100))
print(
    metrics.classification_report(
        tsLbl, predicts.round(), target_names=labelname, digits=4
    )
)

# plot curves on validation loss and accuracy
records = pd.read_csv(folderpath + modelname + ".csv")
plt.figure()
plt.subplot(211)
plt.plot(records["val_loss"], label="validation")
plt.plot(records["loss"], label="training")
plt.yticks([0.00, 0.50, 1.00, 1.50])
plt.title("Loss value", fontsize=12)

ax = plt.gca()
ax.set_xticklabels([])

plt.subplot(212)
plt.plot(records["val_acc"], label="validation")
plt.plot(records["acc"], label="training")
plt.yticks([0.5, 0.6, 0.7, 0.8])
plt.title("Accuracy", fontsize=12)
ax.legend()
plt.show()

plotpath = folderpath + modelname + "_plot.png"
plot_model(
    model, to_file=plotpath, show_shapes=True, show_layer_names=False, rankdir="TB"
)

print("Path to plot:", plotpath)
