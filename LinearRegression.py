from ToolsStructural import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from skrvm import RVR
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import svm

adjs, scores = load_regress_data(['NEO.NEOFAC_N'])
print("The lenght of the dataset is %d" % adjs.shape[0])
adjs_train, adjs_test, scores_train, scores_test = train_test_split(adjs, scores, test_size=0.2, random_state=0)

lin_reg = LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=1)
# 2. Use fit
lin_reg.fit(adjs_train, scores_train)
# 3. Check the score
print(lin_reg.score(adjs_train, scores_train))

scores_pred = lin_reg.predict(X=adjs_test)

print(mean_squared_error(scores_pred, scores_test))

clf = svm.SVR(kernel='rbf', gamma='auto', C=1.0, epsilon=0.1)
clf.fit(adjs_train, scores_train)
print(lin_reg.score(adjs_train, scores_train))
scores_pred = clf.predict(adjs_test)

print(mean_squared_error(scores_pred, scores_test))


clf = RVR(kernel='linear',)
clf.fit(adjs_train, scores_train)
RVR(alpha=1e-06, beta=1e-06, beta_fixed=False, bias_used=True, coef0=0.0, coef1=None, degree=3, kernel='linear',
    n_iter=3000, threshold_alpha=1000000000.0, tol=0.001, verbose=False)
scores_pred = clf.predict(adjs_test)
print(mean_squared_error(scores_pred, scores_test))

def cross_validation():
    pass
'''
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

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mse))'''
