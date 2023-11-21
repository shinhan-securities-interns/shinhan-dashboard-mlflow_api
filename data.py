# # 필요한 부분 import
import pandas as pd
import FinanceDataReader as fdr
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# # DL
import tensorflow as tf

import mlflow
import mlflow.keras

# #################################

#필요한 함수들
# TensorFlow Dataset을 활용한 시퀸스 데이터셋 구성
def windowed_dataset(series, window_size, batch_size, shuffle):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.map(lambda w: (w[:-1], w[-1]))
    return ds.batch(batch_size).prefetch(1)

def preprocess_data(stock_code, window_size=50, batch_size=32):
    # 데이터 수집 및 전처리를 위한 stock_code, window_size, batch_size 인수 사용
    STOCK_CODE = stock_code
    WINDOW_SIZE = window_size
    BATCH_SIZE = batch_size
    
    stock = fdr.DataReader(STOCK_CODE, '2013')
    # IXIC = fdr.DataReader( 'IXIC', '2013')

    scaler = MinMaxScaler()
    scale_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    scaled = scaler.fit_transform(stock[scale_cols])
    df = pd.DataFrame(scaled, columns=scale_cols)

    x_train, x_test, y_train, y_test = train_test_split(df.drop(columns=['Close']), df['Close'], test_size=0.2, random_state=20, shuffle=False)

    train_data = windowed_dataset(y_train, WINDOW_SIZE, BATCH_SIZE, True)
    test_data = windowed_dataset(y_test, WINDOW_SIZE, BATCH_SIZE, False)

    return train_data, test_data, y_test

