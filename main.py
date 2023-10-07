import pandas as pd
from matplotlib import pyplot
import tensorflow as tf
from tensorflow import keras
import numpy as np
# Imported the data set
data = pd.read_csv('household_power_consumption.txt', sep=";")
# print(data)
# Proecess data
NUM_EPOCHS =1
BATCH_SIZE=32

data = np.genfromtxt(
    'household_power_consumption.txt',
    delimiter= ';',
    names=True,
)

featureNames = [
            'Global_active_power',
            'Global_reactive_power',
            'Voltage',
            'Global_intensity',
            'Sub_metering_1',
            'Sub_metering_2',
            'Sub_metering_3'
]
features = []
for featureName in featureNames:
    features.append(data[featureName])
    # print(features)
features = np.stack(features, axis=1)

targets = data['Global_active_power']
# print(targets)
features = features[:-1]
targets = targets[1:]
# print(features)
valid_samples = np.all(np.isfinite(features), axis=1) & np.isfinite(targets)
# valid_samples = np.all(np.isfinite(targets))
features = features[valid_samples]
# targets = targets[valid_samples]
# print(valid_samples)

features = (features-features.mean(axis=0)) / features.std(axis=0)
targets = (targets - targets.mean()) / targets.std()

half_point = int(features.shape[0] // 2)


X_train = features[:half_point]
y_train = targets[:half_point]
X_test = features[half_point:]
y_test = targets[half_point:]
X_train = X_train[:,None,:]
X_test = X_test[:,None,:]
# print(X_train)

model = keras.models.Sequential()
model = keras.models.Sequential(layers=None,name=None)
model.add(keras.layers.LSTM(100))
model.add(keras.layers.Dense(1,activation=None))
model.compile(loss='mse',optimizer="adam")
model.fit(X_train,y_train,epochs=NUM_EPOCHS,batch_size=BATCH_SIZE)

train_predictions = model.predict(X_train,verbose=0)
test_predictions = model.predict(X_test,verbose=0)


fig, axs = pyplot.subplots(nrows=2,dpi=200)
axs[0].plot(y_train[:1000])
axs[0].plot(train_predictions[:1000])
axs[1].plot(y_test[-1000])
axs[1].plot(test_predictions[-1000])
axs[0].set_title("Training Data")
axs[1].set_title("Testing data")
axs[1].legend("True", "predicted")
pyplot.show()
