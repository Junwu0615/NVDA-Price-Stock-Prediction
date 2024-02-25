<a href='https://github.com/Junwu0615/NVDA-Price-Stock-Prediction'><img alt='GitHub Views' src='https://views.whatilearened.today/views/github/Junwu0615/NVDA-Price-Stock-Prediction.svg'> 
<a href='https://github.com/Junwu0615/NVDA-Price-Stock-Prediction'><img alt='GitHub Clones' src='https://img.shields.io/badge/dynamic/json?color=success&label=Clone&query=count_total&url=https://gist.githubusercontent.com/Junwu0615/05f5b34eedbee0ef7d196fdb42ee61f6/raw/NVDA-Price-Stock-Prediction_clone.json&logo=github'> </br>
[![](https://img.shields.io/badge/Project-Stock_Prediction-blue.svg?style=plastic)](https://pse.is/5jtztg) 
[![](https://img.shields.io/badge/Project-Tensorflow_GPU_2.4.0-blue.svg?style=plastic)](https://pypi.org/project/tensorflow/) 
[![](https://img.shields.io/badge/Project-Keras_2.4.3-blue.svg?style=plastic)](https://pypi.org/project/keras/) 
[![](https://img.shields.io/badge/Language-Python_3.8.18-blue.svg?style=plastic)](https://www.python.org/) </br>
[![](https://img.shields.io/badge/Package-Scikit_Learn_1.3.2-green.svg?style=plastic)](https://pypi.org/project/numpy/) 
[![](https://img.shields.io/badge/Package-Numpy_1.19.5-green.svg?style=plastic)](https://pypi.org/project/numpy/) 
[![](https://img.shields.io/badge/Package-Pandas_1.0.0-green.svg?style=plastic)](https://pypi.org/project/pandas/) 
[![](https://img.shields.io/badge/Package-Matplotlib_3.6.0-green.svg?style=plastic)](https://pypi.org/project/matplotlib/) 

## I.　前言
　在 Machine Learning (ML) 火速發展的這幾年，曾幾何時我也非常嚮往從事股票預測 (Stock Prediction) 的研究方向。但這類領域從一開始就有個偽命題，也就是預測價格。它導致了預測結果永遠延遲當前市價步調，無法達到當初人們想用 ML 達成的目的。因此另一個開拓的方向也就誕生了，也就是不預測價格這麼精細的目的，而是預測趨勢 ! 看當下行情是「做多機率多」，亦或是「做空機率多」。扯遠了~ 我閱讀到的這篇文章是經典的預測價格的方法，有幸從網上得到[優質敲門磚](https://www.finlab.tw/%E7%94%A8%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92%E5%B9%AB%E4%BD%A0%E8%A7%A3%E6%9E%90k%E7%B7%9A%E5%9C%96%EF%BC%81/) ~ 我拜讀後並記錄學習過程，另外也為未來定下個目標 [「實現趨勢預測」](https://www.finlab.tw/ml-can-not-predict-price/) !

## II.　如何使用

### STEP.1　CLONE
```py
git clone https://github.com/Junwu0615/NVDA-Price-Stock-Prediction.git
```

### STEP.2　INSTALL PACKAGES
#### 用 pip 安裝所需套件
```py
pip install -r requirements.txt
```

#### 或是直接安裝 Conda 環境檔 ( yaml 需自行設定使用者變數名稱: `xxx` -> `?` )
```py
conda env create -f C:\Users\xxx\environment.yaml
```

###  STEP.3　RUN
```py
python NVDA-Price-Stock-Prediction.py -h
```

### STEP.4　HELP
- `-h` Help : Show this help message and exit.
- `-open` Open :　Add open feature ?　`T` / `F`
- `-high` High :　Add high feature ?　`T` / `F`
- `-low` Low :　Add low feature ?　`T` / `F`
- `-vol` Volume :　Add volume feature ?　`T` / `F`
- `-utm` :　Use Trained Model ?　`T` / `F`

### STEP.5　EXAMPLE
- 使用「開盤價」、「最高價」、「最低價」、「交易量」， 4 種特徵來預測「收盤價」。
```py
python NVDA-Price-Stock-Prediction.py -open T -high T -low T -vol T -utm F
```
- <img width='600' height='450' src="https://github.com/Junwu0615/NVDA-Price-Stock-Prediction/blob/main/output/output_model_open_high_low_volume.png"/>

- 只使用「交易量」此特徵來預測「收盤價」。
```py
python NVDA-Price-Stock-Prediction.py -open F -high F -low F -vol T -utm F
```
- <img width='600' height='450' src="https://github.com/Junwu0615/NVDA-Price-Stock-Prediction/blob/main/output/output_model_volume.png"/>

- 使用已訓練過的模型進行預測。
```py
python NVDA-Price-Stock-Prediction.py -open F -high F -low F -vol T -utm T
```

## III.　說明過程

### A.　資料取得
我透過 [Financial Modeling Prep](https://site.financialmodelingprep.com/) 取得我想預測的標的 NVIDIA ，它需要註冊會員以取得 API key，接著使用日線圖的 API 叫出該 Symbol 歷史資料。另外，我已先將其內容存儲成 json 格式 -> [`NVDA_history_20190225_20240223.json`](/dataset/NVDA_history_20190225_20240223.json)。
- <img width='650' height='500' src="https://github.com/Junwu0615/NVDA-Price-Stock-Prediction/blob/main/output/NVDA_History_Price.png"/>

### B.　資料預處理
- 原始資料如下圖所示。
- <img width='400' height='450' src="https://github.com/Junwu0615/NVDA-Price-Stock-Prediction/blob/main/sample_img/origin.jpg"/>
- 將資料範圍大小，規範於 0 至 1 區間，標準化後的資料如下圖所示。
- <img width='430' height='450' src="https://github.com/Junwu0615/NVDA-Price-Stock-Prediction/blob/main/sample_img/norm.jpg"/>
- 接著將資料切割成訓練集 ( 90% ) 和測試集 ( 10% )，任務目標是預測「收盤價」，而其餘特徵都是餵給機器的輸入。
  ```
  x_train = result[:int(number_train), :-1]
  y_train = result[:int(number_train), -1][:,-1]
  ```

### C.　模型建立與訓練
- 模型使用序列神經模型的 GRU (原先文章使用 LSTM)，實際跑過一遍後，效果確實也比較好。
- 我有測試幾組實驗，心得是使用雙層神經網路，且第一層維度設置不能比第二層小 (1024, 1024)。
- 丟失率 `0.6` 左右比較好，可以自行嘗試看看。批量大小依照原先設定 `128`，epoch 設定 `200`。
- 都設置好即可以開始訓練模型 ! (我起初玩 ML 流程都沒有這麼短...哩哩叩叩要設定一堆)

### D.　預測結果與真實差異
- 將結果還回正常區間後，用 Matplotlib 檢視成果 ( 因為不可能預測得到精確的數字，所以用趨勢的方式來檢視 )。 可參考 `STEP.5　EXAMPLE`。另外有趣的是用單一特徵反而比多特徵效果還來的好，不知什麼原因 @@?

## IV.　參考資源
- [FinLab | 用深度學習幫你解析K線圖！](https://www.finlab.tw/%E7%94%A8%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92%E5%B9%AB%E4%BD%A0%E8%A7%A3%E6%9E%90k%E7%B7%9A%E5%9C%96%EF%BC%81/)
- [Financial Modeling Prep | 預測資料來源](https://site.financialmodelingprep.com/)