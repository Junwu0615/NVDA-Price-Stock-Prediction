import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from argparse import ArgumentParser
from sklearn import preprocessing
import tensorflow as tf
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Dropout, Activation

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def parse_args():
    parse = ArgumentParser()
    parse.add_argument("-open", "--open", help = "Add open feature ?", default = "T", type = str)
    parse.add_argument("-high", "--high", help = "Add high feature ?", default = "T", type = str)
    parse.add_argument("-low", "--low", help = "Add low feature ?", default = "T", type = str)
    parse.add_argument("-vol", "--volume", help = "Add volume feature ?", default = "T", type = str)
    parse.add_argument("-utm", "--use_trained_model", help = "Use Trained Model ?", default = "F", type = str)
    args = parse.parse_args()
    return args

class stock_prediction:
    def __init__(self):
        file = "./dataset/NVDA_history_20190225_20240223.json"
        with open(file, "r") as f:
            for i in f: self.jsfile = json.loads(i)
        f.close()
        
    def get_need_features(self, open_, high_, low_, volume_):
        self.df = pd.DataFrame(self.jsfile)
        total_ = []; title = []; 
        if open_ == "T" or open_ == "True" or open_ == "t": 
            title.append("open")
        if high_ == "T" or high_ == "True" or high_ == "t": 
            title.append("high")
        if low_ == "T" or low_ == "True" or low_ == "t": 
            title.append("low")
        if volume_ == "T" or volume_ == "True" or volume_ == "t": 
            title.append("volume")
        title.append("close")
        for i in sorted(list(self.df.columns)):
            temp = []
            l = list(self.df[i])
            if open_ == "T" or open_ == "True" or open_ == "t": 
                temp.append(l[0])
            if high_ == "T" or high_ == "True" or high_ == "t": 
                temp.append(l[1])
            if low_ == "T" or low_ == "True" or low_ == "t": 
                temp.append(l[2])
            if volume_ == "T" or volume_ == "True" or volume_ == "t": 
                temp.append(l[5])
            temp.append(l[3])
            total_.append(temp)
        self.new_df = pd.DataFrame(total_)
        self.new_df.columns = title
    
    def origin_status(self, open_, high_, low_, volume_):
        t_list = []
        for i in range(11):
            if i == 0: t_list.append(sorted(list(self.df.columns))[i])
            else: t_list.append(sorted(list(self.df.columns))[125*i])
        fig, ax1 = plt.subplots(figsize=(12, 10))
        plt.title('NVDA History Price')
        plt.xlabel('Time')
        ax2 = ax1.twinx()
        ax1.set_ylabel('Price')
        if open_ == "T" or open_ == "True" or open_ == "t": 
            ax1.plot(list(self.new_df["open"]),color='black', label='Open Price')
        if high_ == "T" or high_ == "True" or high_ == "t": 
            ax1.plot(list(self.new_df["high"]),color='red', label='High Price')
        if low_ == "T" or low_ == "True" or low_ == "t": 
            ax1.plot(list(self.new_df["low"]),color='green', label='Low Price')
        
        ax1.plot(list(self.new_df["close"]),color='blue', label='Close Price')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.legend(loc='upper left')
        
        if volume_ == "T" or volume_ == "True" or volume_ == "t": 
            ax2.set_ylabel('Volume')
            ax2.plot(list(self.new_df["volume"]),'m--', label='Volume')
            ax2.tick_params(axis='y', labelcolor='black')
            ax2.legend(loc='upper right')
        
        ax1.tick_params(axis='x', labelrotation=90)
        plt.savefig("./output/NVDA_History_Price.png")
        plt.clf()
    
    def normalize(self):
        df_norm = self.new_df.copy()
        min_max_scaler = preprocessing.MinMaxScaler()
        if open_ == "T" or open_ == "True" or open_ == "t": 
            df_norm['open'] = min_max_scaler.fit_transform(self.new_df.open.values.reshape(-1,1))
        if high_ == "T" or high_ == "True" or high_ == "t": 
            df_norm['high'] = min_max_scaler.fit_transform(self.new_df.high.values.reshape(-1,1))
        if low_ == "T" or low_ == "True" or low_ == "t": 
            df_norm['low'] = min_max_scaler.fit_transform(self.new_df.low.values.reshape(-1,1))
        if volume_ == "T" or volume_ == "True" or volume_ == "t": 
            df_norm['volume'] = min_max_scaler.fit_transform(self.new_df.volume.values.reshape(-1,1))
        df_norm['close'] = min_max_scaler.fit_transform(self.new_df.close.values.reshape(-1,1))
        return df_norm
    
    def data_helper(self, df_norm, time_frame):
        # 資料維度: 開盤價、收盤價、最高價、最低價、成交量, 5維
        number_features = len(df_norm.columns)
        
        # 將dataframe 轉成 numpy array
        datavalue = df_norm.values
        
        result = []
        # 若想要觀察的 time_frame 為20天, 需要多加一天做為驗證答案
        for index in range( len(datavalue) - (time_frame+1) ): # 從 datavalue 的第0個跑到倒數第 time_frame+1 個
            result.append(datavalue[index: index + (time_frame+1) ]) # 逐筆取出 time_frame+1 個K棒數值做為一筆 instance
            
        result = np.array(result)
        number_train = round(0.9 * result.shape[0]) # 取 result 的前90% instance做為訓練資料
        
        x_train = result[:int(number_train), :-1] # 訓練資料中, 只取每一個 time_frame 中除了最後一筆的所有資料做為feature
        y_train = result[:int(number_train), -1][:,-1] # 訓練資料中, 取每一個 time_frame 中最後一筆資料的最後一個數值(收盤價)做為答案
        
        # 測試資料
        x_test = result[int(number_train):, :-1]
        y_test = result[int(number_train):, -1][:,-1]
    
        # 將資料組成變好看一點
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], number_features))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], number_features))  
    
        return [x_train, y_train, x_test, y_test]
    
    def build_model(self, input_length, input_dim):
        d = 0.6
        model = Sequential()
        
        model.add(GRU(1024, input_shape=(input_length, input_dim), return_sequences=True))
        model.add(Dropout(d))
    
        model.add(GRU(1024, input_shape=(input_length, input_dim), return_sequences=False))
        model.add(Dropout(d))
    
        model.add(Dense(16,kernel_initializer="uniform",activation='relu'))
        model.add(Dense(1,kernel_initializer="uniform",activation='linear'))
        model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
    
        return model
    
    def denormalize(self, norm_value):
        original_value = self.new_df['close'].values.reshape(-1,1)
        norm_value = norm_value.reshape(-1,1)
        min_max_scaler = preprocessing.MinMaxScaler()
        min_max_scaler.fit_transform(original_value)
        denorm_value = min_max_scaler.inverse_transform(norm_value)
        return denorm_value
    
    
    def show_result(self, denorm_pred, denorm_ytest, model_name):
        plt.figure(figsize=(8,6))
        plt.plot(denorm_pred, color='red', label='Prediction')
        plt.plot(denorm_ytest, color='blue', label='Answer')
        plt.legend(loc='best')
        #plt.show()
        plt.savefig(f"./output/output_{model_name}.png")
        plt.clf()

if __name__ == "__main__":
    # Get parameters
    args = parse_args()
    use_trained_model = args.use_trained_model
    open_ = args.open
    high_ = args.high
    low_ = args.low
    volume_ = args.volume
    if use_trained_model == "T" or use_trained_model == "True" or use_trained_model == "t": 
        use_trained_model = True
    else: use_trained_model = False

    sp = stock_prediction()
    
    # feed features to create data frame
    sp.get_need_features(open_, high_, low_, volume_)
    
    # show NVDA history price of status
    sp.origin_status(open_, high_, low_, volume_) 
    
    # 標準化數值
    df_norm = sp.normalize()
    
    # 以20天為一區間進行股價預測
    x_train, y_train, x_test, y_test = sp.data_helper(df_norm, 1)
    
    # 20天、5維
    model = sp.build_model(1, len(df_norm.columns)) 
    
    # 模型名稱命名
    model_name = "model"
    if open_ == "T" or open_ == "True" or open_ == "t": 
        model_name += "_open"
    if high_ == "T" or high_ == "True" or high_ == "t": 
        model_name += "_high"
    if low_ == "T" or low_ == "True" or low_ == "t": 
        model_name += "_low"
    if volume_ == "T" or volume_ == "True" or volume_ == "t": 
        model_name += "_volume"
    
    # 一個batch有128個instance，總共跑50個迭代
    if use_trained_model == True:
        model = tf.keras.models.load_model(f"./trained_model/{model_name}.keras")
    else:
        model.fit(x_train, y_train, batch_size=128, epochs=200, validation_split=0.1, verbose=1)
        model.save(f"./trained_model/{model_name}.keras")
        
    # 用訓練好的 LSTM 模型對測試資料集進行預測
    pred = model.predict(x_test) 
    
    # 將預測值與正確答案還原回原來的區間值
    denorm_pred = sp.denormalize(pred)
    denorm_ytest = sp.denormalize(y_test)
    
    # 用趨勢圖來呈現結果
    sp.show_result(denorm_pred, denorm_ytest, model_name)