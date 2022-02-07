import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import adam_v2
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.utils import set_random_seed

set_random_seed(42)


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)


def annotate_bars(ax, horizontal=True, scale=1):
    for p in ax.patches:
        if horizontal:
            xpos = p.get_width() + p.get_width() * 0.01 * scale
            ax.annotate(f"{round(p.get_width(), 3)}", (xpos, p.get_y() + 0.1), fontsize=14)
        else:
            ypos = p.get_height() + p.get_height() * 0.01 * scale
            ax.annotate(f"{round(p.get_height(), 3)}", (p.get_x() + p.get_width() / 2, ypos), fontsize=14)
    return ax


def plot_performance(reports, features=["MAPE", "RMSE", "R2"]):
    plt.figure(figsize=(16, 9))
    performance = pd.DataFrame(reports).T[features]
    ax = performance.plot(kind="barh", figsize=(16, 9), fontsize=16)
    ax = annotate_bars(ax, horizontal=True)
    plt.show()


def evaluation_report(y, yhat):
    mape = mean_absolute_percentage_error(y, yhat)
    rmse = np.sqrt(mean_squared_error(y, yhat))
    r2 = r2_score(y, yhat)
    return mape, rmse, r2


def rmse(y, yhat):
    return np.sqrt(mean_squared_error(y, yhat))


def parse_data(X, y):
    n = len(X) // 10
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=n, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=n, random_state=42)

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    X_dev_scaled = scaler.transform(X_dev)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), (X_dev, y_dev), (X_dev_scaled, X_test_scaled)


def build_model(shape=8, lr=0.0001, activation="relu", out_activation=None):
    x = Input(shape=(shape))
    h = Dense(32, activation=activation)(x)
    h = Dense(16, activation=activation)(h)
    h = Dense(8, activation=activation)(h)
    h = Dense(4, activation=activation)(h)
    h = Dense(2, activation=activation)(h)
    out = Dense(1, activation=out_activation)(h)

    model = Model(inputs=x, outputs=out)

    lr_schedule = ExponentialDecay(
        lr,
        decay_steps=10000,
        decay_rate=0.96,
        staircase=True)

    model.compile(loss="mse", optimizer=adam_v2.Adam(learning_rate=lr_schedule))
    return model


def calc_rmse(y, yhat):
    return np.sqrt(mean_squared_error(y, yhat))


def evaluate_model(model, X, y, scaler=None):
    y_pred = model.predict(X)
    if scaler is not None:
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y = scaler.inverse_transform(y.reshape(-1, 1)).flatten()
    return r2_score(y, y_pred), calc_rmse(y, y_pred)


def te_report(model,
              X_train, y_train,
              X_val, y_val,
              X_test, y_test):
    results = []
    for name, (X_, y_) in zip(["Train", "Val.", "Test"], [(X_train, y_train), (X_val, y_val), (X_test, y_test)]):
        r2, rmse = evaluate_model(model, X_, y_)
        results.append([name, r2, rmse])
    results_df = pd.DataFrame(results, columns=["Data", "R2", "RMSE"])
    return results_df
