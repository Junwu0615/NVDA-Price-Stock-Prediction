# -*- coding: utf-8 -*-
"""
@author: PC
Update Time: 2024-11-24
"""

import json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import tensorflow as tf
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Dropout, Activation

class Model:
    def __init__(self, obj):
        self.open = obj.open
        self.high = obj.high
        self.low = obj.low
        self.volume = obj.volume
        self.use_trained_model = obj.use_trained_model
        self.norm_check = ['T', 'True', 't']

        file = "./datasets/NVDA_history_20190225_20240223.json"
        self.loader = [json.loads(i) for i in open(file, "r")][0]

        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

    def check_folder(self, path: str):
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)

    def get_need_features(self):
        self.df = pd.DataFrame(self.loader)
        predata, self.col, idx = [], [], []
        if self.open in self.norm_check:
            self.col.append("open")
            idx += [0]
        if self.high in self.norm_check:
            self.col.append("high")
            idx += [1]
        if self.low in self.norm_check:
            self.col.append("low")
            idx += [2]
        if self.volume in self.norm_check:
            self.col.append("volume")
            idx += [5]
        self.col.append("close")
        for i in sorted(list(self.df.columns)):
            l = list(self.df[i])
            temp = [l[ix] for ix in idx]
            temp.append(l[3])
            predata.append(temp)
        self.new_df = pd.DataFrame(predata)
        self.new_df.columns = self.col

    def origin_status(self):
        t_list = []
        for i in range(11):
            t_list += sorted(list(self.df.columns))[i] if i == 0 \
                else sorted(list(self.df.columns))[125 * i]
        fig, ax1 = plt.subplots(figsize=(12, 10))
        plt.title('NVDA History Price')
        plt.xlabel('Time')
        ax2 = ax1.twinx()
        ax1.set_ylabel('Price')

        for cl in self.col:
            if cl != ["volume"]:
                ax1.plot(list(self.new_df[cl]), label=f'{cl.capitalize()} Price')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.legend(loc='upper left')

        if self.volume in self.norm_check:
            ax2.set_ylabel('Volume')
            ax2.plot(list(self.new_df["volume"]), 'm--', label='Volume')
            ax2.tick_params(axis='y', labelcolor='black')
            ax2.legend(loc='upper right')

        ax1.tick_params(axis='x', labelrotation=90)
        plt.savefig("./sample/NVDA_History_Price.png")
        plt.clf()

    def normalize(self) -> pd.DataFrame:
        df_norm = self.new_df.copy()
        min_max_scaler = preprocessing.MinMaxScaler()
        for cl in self.col:
            df_norm[cl] = min_max_scaler.fit_transform(self.new_df[cl].values.reshape(-1, 1))
        return df_norm

    def data_helper(self, df_norm, time_frame) -> list:
        # 資料維度: 開盤價、收盤價、最高價、最低價、成交量 # 五維
        number_features = len(df_norm.columns)

        # 將dataframe 轉成 numpy array
        dv = df_norm.values

        result = []
        # 若想要觀察的 time_frame 為20天, 需要多加一天做為驗證答案
        # 從 dv 的第0個跑到倒數第 time_frame+1 個
        for index in range(len(dv) - (time_frame + 1)):
            # 逐筆取出 time_frame+1 個K棒數值做為一筆 instance
            result.append(dv[index: index + (time_frame + 1)])

        result = np.array(result)

        # 取 result 的前90% instance做為訓練資料
        number_train = round(0.9 * result.shape[0])

        # 訓練資料中, 只取每一個 time_frame 中除了最後一筆的所有資料做為feature
        x_train = result[:int(number_train), :-1]

        # 訓練資料中, 取每一個 time_frame 中最後一筆資料的最後一個數值(收盤價)做為答案
        y_train = result[:int(number_train), -1][:, -1]

        # 測試資料
        x_test = result[int(number_train):, :-1]
        y_test = result[int(number_train):, -1][:, -1]

        # 將資料組成變好看一點
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], number_features))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], number_features))

        return [x_train, y_train, x_test, y_test]

    def build_model(self, input_length, input_dim) -> Sequential:
        d = 0.6
        model = Sequential()

        model.add(GRU(1024, input_shape=(input_length, input_dim), return_sequences=True))
        model.add(Dropout(d))

        model.add(GRU(1024, input_shape=(input_length, input_dim), return_sequences=False))
        model.add(Dropout(d))

        model.add(Dense(16, kernel_initializer="uniform", activation='relu'))
        model.add(Dense(1, kernel_initializer="uniform", activation='linear'))
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        return model

    def denormalize(self, norm_value) -> object:
        original_value = self.new_df['close'].values.reshape(-1, 1)
        norm_value = norm_value.reshape(-1, 1)
        min_max_scaler = preprocessing.MinMaxScaler()
        min_max_scaler.fit_transform(original_value)
        denorm_value = min_max_scaler.inverse_transform(norm_value)
        return denorm_value

    def make_img(self, denorm_pred, denorm_ytest, model_name):
        plt.figure(figsize=(8, 6))
        plt.plot(denorm_pred, color='red', label='Prediction')
        plt.plot(denorm_ytest, color='blue', label='Answer')
        plt.legend(loc='best')
        # plt.show()
        plt.savefig(f"./sample/output_{model_name}.png")
        plt.clf()

    def main(self):
        self.check_folder('./sample')
        self.check_folder('./trained')

        if self.use_trained_model in ['T', 'True', 't']:
            self.use_trained_model = True
        else:
            self.use_trained_model = False

        # feed features to create data frame
        self.get_need_features()

        # show NVDA history price of status
        self.origin_status()

        # 標準化數值
        df_norm = self.normalize()

        # 以20天為一區間進行股價預測
        x_train, y_train, x_test, y_test = self.data_helper(df_norm, 1)

        # 20天、5維
        model = self.build_model(1, len(df_norm.columns))

        # 模型名稱命名
        model_name = "model"
        if self.open in self.norm_check:
            model_name += "_open"
        if self.high in self.norm_check:
            model_name += "_high"
        if self.low in self.norm_check:
            model_name += "_low"
        if self.volume in self.norm_check:
            model_name += "_volume"

        # 一個batch有128個instance，總共跑50個迭代
        if self.use_trained_model:
            model = tf.keras.models.load_model(f"./trained/{model_name}.keras")
        else:
            model.fit(x_train, y_train, batch_size=128, epochs=200, validation_split=0.1, verbose=1)
            model.save(f"./trained/{model_name}.keras")

        # 用訓練好的 LSTM 模型對測試資料集進行預測
        pred = model.predict(x_test)

        # 將預測值與正確答案還原回原來的區間值
        denorm_pred = self.denormalize(pred)
        denorm_ytest = self.denormalize(y_test)

        # 用趨勢圖來呈現結果
        self.make_img(denorm_pred, denorm_ytest, model_name)
