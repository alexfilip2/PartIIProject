from ToolsStructural import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Sperate train and test data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import svm

adjs, scores = load_regress_data(['NEO.NEOFAC_A'])
print(adjs.shape)
print(scores.shape)
X_train, X_test, y_train, y_test = train_test_split(adjs, scores, test_size=0.2, random_state=0)

model = LinearRegression()
# 2. Use fit
model.fit(X_train, y_train)
# 3. Check the score
print(model.score(X_test, y_test))

y_pred = model.predict(X_test)

print(mean_squared_error(y_pred, np.squeeze(y_test)))

clf = svm.SVR()
clf.fit(X_train, np.squeeze(y_train))

y_pred = clf.predict(X_test)

print(mean_squared_error(y_pred, np.squeeze(y_test)))


def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[X_train.shape[-1]]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
    ])

    optimizer = tf.train.RMSPropOptimizer(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


model = build_model()


# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')


EPOCHS = 10000

history = model.fit(
    X_train, np.squeeze(y_train),
    epochs=EPOCHS, validation_split=0.2, verbose=0,
    callbacks=[PrintDot()])
history = history.history

import matplotlib.pyplot as plt


def plot_history(hist):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epochs'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epochs'], hist['val_mean_absolute_error'],
             label='Val Error')
    plt.legend()
    plt.ylim([0, 5])

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epochs'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epochs'], hist['val_mean_squared_error'],
             label='Val Error')
    plt.legend()
    plt.ylim([0, 20])


loss, mae, mse = model.evaluate(X_test, np.squeeze(y_test), verbose=0)

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mse))
