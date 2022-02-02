import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Conv1D, Dense, Flatten, MaxPool1D
from keras.models import Model
from keras.optimizers import adam_v2
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)


def construct_continent_model(x, activation="relu"):
    c = Conv1D(64, 4, strides=2, padding="same", name="Conv1D_1_c")(x)
    c = MaxPool1D(name="MaxPooling_1_c", padding="same")(c)
    c = Conv1D(32, 4, strides=1, padding="same", name="Conv1D_2")(c)
    c = MaxPool1D(name="MaxPooling_2_c", padding="same")(c)
    c = Flatten(name="Flatten_c")(c)

    h = Dense(64, activation=activation, name="Hiddenlayer1_c")(c)
    h = Dense(32, activation=activation, name="Hiddenlayer2_c")(h)

    fc = Dense(32, activation=activation, name="Hiddenlayer3_local")(h)
    fc = Dense(8, activation=activation, name="Hiddenlayer4_local")(fc)
    fc = Dense(4, activation=activation, name="Hiddenlayer5_local")(fc)
    fc = Dense(2, activation=activation, name="Hiddenlayer6_local")(fc)
    out = Dense(1, name="Hiddenlayer7_local")(fc)

    model = Model(inputs=x, outputs=out)
    return model, h


def construct_local_model(x, activation="relu"):
    c = Conv1D(32, 4, strides=2, padding="same", name="Conv1D_1")(x)
    c = Conv1D(32, 4, strides=1, padding="same", name="Conv1D_2")(c)
    c = MaxPool1D(name="MaxPooling", padding="same")(c)
    c = Flatten(name="Flatten")(c)

    h = Dense(512, activation=activation, name="Hiddenlayer1_local")(c)
    h = Dense(64, activation=activation, name="Hiddenlayer2_local")(h)
    h = Dense(32, activation=activation, name="Hiddenlayer3_local")(h)

    fc = Dense(32, activation=activation, name="Hiddenlayer4_local")(h)
    fc = Dense(16, activation=activation, name="Hiddenlayer5_local")(fc)
    fc = Dense(5, activation=activation, name="Hiddenlayer6_local")(fc)
    fc = Dense(2, activation=activation, name="Hiddenlayer7_local")(fc)
    out = Dense(1, name="Hiddenlayer8_local")(fc)

    model = Model(inputs=x, outputs=out)
    return model, h


def fit_model(model, X_train, y_train, X_val, y_val, bs=10000, lr=0.001, epochs=400):
    es = EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=30,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    )
    model.compile(loss="mse", optimizer=adam_v2.Adam(lr))
    return model.fit(X_train, y_train, callbacks=[es], validation_data=(X_val, y_val), epochs=epochs, batch_size=bs)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def get_error(y_true, y_pred):
    # return np.square(y_true - y_pred)
    return y_true - y_pred


class Utils:
    def __init__(self, path):
        self.path = path
        f = np.load(path, allow_pickle=True)
        self.X_train = f["X_train"]
        self.y_train = f["y_train"] / 1e6
        self.cont_train = f["cont_train"]

        self.X_test = f["X_train"]
        self.y_test = f["y_train"] / 1e6
        self.cont_test = f["cont_test"]
        self.f = f

    def get_train_val(self, continent=0):
        idx = np.where(self.cont_train == continent)[0]
        x_ = self.X_train[idx, :]
        y_ = self.y_train[idx]
        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            x_, y_, test_size=0.1, random_state=42)

        scaler = MinMaxScaler()
        scaler.fit(X_train_raw)

        X_train = scaler.transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)

        return X_train, y_train, X_val, y_val, scaler

    def train_model(self, model, continent, bs=10000, lr=0.001, epochs=400):
        X_train, y_train, X_val, y_val, scaler = self.get_train_val(continent)
        return fit_model(model, X_train, y_train, X_val, y_val, bs=bs, lr=lr, epochs=epochs)

    def predict_test(self, model, continent):
        _, _, _, _, scaler = self.get_train_val(continent)

        idx = np.where(self.cont_test == continent)[0]
        x_ = self.X_test[idx, :]
        y_true = self.y_test[idx]
        x = scaler.transform(x_)

        y_pred = model.predict(x).flatten()
        return y_true, y_pred

    def predict_train_val(self, model, continent):
        X_train, y_train, X_val, y_val, scaler = self.get_train_val(continent)
        y_pred_train = model.predict(X_train).flatten()
        y_pred_val = model.predict(X_val).flatten()

        return y_train, y_pred_train, y_val, y_pred_val
