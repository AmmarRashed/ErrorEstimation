from joblib import Parallel, delayed
from keras.callbacks import EarlyStopping
from scipy.stats import ttest_ind
from sklearn.ensemble import RandomForestRegressor

from utils import *


class ErrorCorrection:
    def __init__(self, X, y, te_model=None):
        (self.X_train, self.y_train), \
        (self.X_val, self.y_val), \
        (self.X_test, self.y_test), \
        (self.X_dev, self.y_dev), \
        (self.X_dev_scaled, self.X_test_scaled) = parse_data(X, y)

        self.te_model = build_model(shape=X.shape[1]) if te_model is None else te_model

    def train_te(self):
        es = EarlyStopping(patience=100, restore_best_weights=True)
        H = self.te_model.fit(x=self.X_train, y=self.y_train, validation_data=(self.X_val, self.y_val), epochs=10000,
                              batch_size=len(self.X_train),
                              callbacks=[es])
        return H

    def te_performance(self):
        return te_report(self.te_model, self.X_train, self.y_train, self.X_val, self.y_val,
                         self.X_test_scaled, self.y_test)

    def error_estimation(self):
        yhat_dev = self.te_model.predict(self.X_dev_scaled).flatten()
        yhat_test = self.te_model.predict(self.X_test_scaled).flatten()

        e_dev = self.y_dev - yhat_dev
        e_test = self.y_test - yhat_test

        X_dev_yhat = np.concatenate((self.X_dev, yhat_dev.reshape(-1, 1)), axis=1)
        X_test_yhat = np.concatenate((self.X_test, yhat_test.reshape(-1, 1)), axis=1)

        ee_model = RandomForestRegressor(n_estimators=250, n_jobs=-1, random_state=42)
        ee_model.fit(X_dev_yhat, e_dev)
        e_test_hat = ee_model.predict(X_test_yhat)

        corrected_yhat = e_test_hat + yhat_test
        results = pd.DataFrame([evaluation_report(self.y_test, yhat_test),
                                evaluation_report(self.y_test, corrected_yhat)], columns=["MAPE", "RMSE", "R2"],
                               index=["Before", "After"])
        old = np.abs((self.y_test - yhat_test) / self.y_test)
        new = np.abs((self.y_test - corrected_yhat) / self.y_test)
        t = ttest_ind(new, old, alternative='less')
        print(f"P-Value: {t.pvalue}")
        return results
