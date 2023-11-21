import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import FinanceDataReader as fdr

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

# DL
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Conv1D,LSTM, Dense
from keras.optimizers import Adam

from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping

import mlflow
import mlflow.keras

################################################################
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

def create_mlflow_experiment(mlflow_uri, experiment_name):
    mlflow.set_tracking_uri(mlflow_uri)
    exp_id = mlflow.create_experiment(experiment_name)
    return exp_id



## 1.데이터 전처리 함수 정의
def preprocess_data(stock_code, window_size=50, batch_size=32):
    # 데이터 수집 및 전처리를 위한 stock_code, window_size, batch_size 인수 사용
    STOCK_CODE = stock_code
    WINDOW_SIZE = window_size
    BATCH_SIZE = batch_size
    
    stock = fdr.DataReader(STOCK_CODE, '2013')

    scaler = MinMaxScaler()
    scale_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    scaled = scaler.fit_transform(stock[scale_cols])
    df = pd.DataFrame(scaled, columns=scale_cols)

    x_train, x_test, y_train, y_test = train_test_split(df.drop(columns=['Close']), df['Close'], test_size=0.2, random_state=20, shuffle=False)

    train_data = windowed_dataset(y_train, WINDOW_SIZE, BATCH_SIZE, True)
    test_data = windowed_dataset(y_test, WINDOW_SIZE, BATCH_SIZE, False)

    return train_data, test_data


# 2. MLflow 초기화 및 추적 URI 설정
mlflow_uri = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(mlflow_uri)


# 3. 실험 생성 및 실험 ID 얻기
experiment_name = "prediction"
exp_id = create_mlflow_experiment(mlflow_uri, experiment_name)

# 주식 코드
stock_code = 'KS11'

# 4. 각 모델 훈련 루프 실행
for i in range(1, 2):
    # 1번째 모델
    with mlflow.start_run(experiment_id=exp_id, run_name = 'dl_autolog_1'):
        mlflow.keras.autolog()

        # Preprocess the data using the provided stock code
        WINDOW_SIZE = 50
        BATCH_SIZE = 32
        train_data, test_data = preprocess_data(stock_code, WINDOW_SIZE, BATCH_SIZE)

        # Model design and compilation
        model = Sequential([
            Conv1D(filters=64, kernel_size=10, padding="same", activation="relu", input_shape=[WINDOW_SIZE, 1]),
            LSTM(16, activation='tanh'),
            Dense(16, activation="relu"),
            Dense(1)
        ])
        model.compile(loss=Huber(), optimizer=Adam(0.0005))

        # Early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)

        # Training
        history = model.fit(train_data, validation_data=test_data, epochs=50, callbacks=[early_stopping]).history

        # Log the model
        mlflow.keras.log_model(model, "model", registered_model_name= "stock_prediction")

        # Disable autolog
        mlflow.keras.autolog(disable=True)


    ######################################
    # 2 번째 모델  
    with mlflow.start_run(experiment_id=exp_id, run_name='dl_autolog_2'):
        
        mlflow.keras.autolog()

        # Preprocess the data using the provided stock code
        WINDOW_SIZE = 50
        BATCH_SIZE = 32
        train_data, test_data = preprocess_data(stock_code, WINDOW_SIZE, BATCH_SIZE)

    #모델 설계 및 컴파일
        model2 = Sequential([
            # 1차원 feature map 생성
            Conv1D(filters=64, kernel_size=5,
            padding="same",
            activation="relu",
            input_shape=[WINDOW_SIZE, 1]),
            # LSTM
            LSTM(16, activation='tanh'),
            Dense(16, activation="relu"),
            Dense(1) ])
        # Sequence 학습에 비교적 좋은 퍼포먼스를 내는 Huber()를 사용합니다.
        loss = Huber()
        optimizer = Adam(0.0005)
        model2.compile(loss=Huber(), optimizer= optimizer)
        # earlystopping은 10번 epoch통안 val_loss 개선이 없다면 학습을 멈춥니다.
        earlystopping = EarlyStopping(monitor='val_loss', patience=10)
    #학습
        history = model2.fit(train_data,
                        validation_data=(test_data),
                        epochs=50,
                        callbacks=[earlystopping]).history
    #모델 등록 (best 모델 기록하기)
        mlflow.keras.log_model(model2, "model2",  registered_model_name="stock_prediction")
    #autolog 종료
        mlflow.keras.autolog(disable = True)


    ######################################
    # 3 번째 모델  
    with mlflow.start_run(experiment_id=exp_id, run_name='dl_autolog_3'):
        
        mlflow.keras.autolog()

        # Preprocess the data using the provided stock code
        WINDOW_SIZE = 50
        BATCH_SIZE = 32
        train_data, test_data = preprocess_data(stock_code, WINDOW_SIZE, BATCH_SIZE)

    #모델 설계 및 컴파일
        model3 = Sequential([
            # 1차원 feature map 생성
            Conv1D(filters=32, kernel_size=10,
            padding="same",
            activation="relu",
            input_shape=[WINDOW_SIZE, 1]),
            # LSTM
            LSTM(16, activation='tanh'),
            Dense(16, activation="relu"),
            Dense(1) ])
        # Sequence 학습에 비교적 좋은 퍼포먼스를 내는 Huber()를 사용합니다.
        loss = Huber()
        optimizer = Adam(0.0005)
        model3.compile(loss=Huber(), optimizer= optimizer)
        # earlystopping은 10번 epoch통안 val_loss 개선이 없다면 학습을 멈춥니다.
        earlystopping = EarlyStopping(monitor='val_loss', patience=10)
    #학습
        history = model3.fit(train_data,
                        validation_data=(test_data),
                        epochs=50,
                        callbacks=[earlystopping]).history
    #모델 등록 (best 모델 기록하기)
        mlflow.keras.log_model(model3, "model3",  registered_model_name="stock_prediction")
    #autolog 종료
        mlflow.keras.autolog(disable = True)


    ######################################
    # 4 번째 실행   
    with mlflow.start_run(experiment_id=exp_id, run_name='dl_autolog_4'):
        
        mlflow.keras.autolog()

        # Preprocess the data using the provided stock code
        WINDOW_SIZE = 50
        BATCH_SIZE = 32
        train_data, test_data = preprocess_data(stock_code, WINDOW_SIZE, BATCH_SIZE)

    #모델 설계 및 컴파일
        model4 = Sequential([
            # 1차원 feature map 생성
            Conv1D(filters=64, kernel_size=10,
            padding="same",
            activation="relu",
            input_shape=[WINDOW_SIZE, 1]),
            # LSTM
            LSTM(16, activation='tanh'),
            Dense(32, activation="relu"),
            Dense(16, activation="relu"),
            Dense(8, activation="relu"),
            Dense(1) ])
        # Sequence 학습에 비교적 좋은 퍼포먼스를 내는 Huber()를 사용합니다.
        loss = Huber()
        optimizer = Adam(0.0005)
        model4.compile(loss=Huber(), optimizer= optimizer)
        # earlystopping은 10번 epoch통안 val_loss 개선이 없다면 학습을 멈춥니다.
        earlystopping = EarlyStopping(monitor='val_loss', patience=10)
    #학습
        history = model4.fit(train_data,
                        validation_data=(test_data),
                        epochs=50,
                        callbacks=[earlystopping]).history
    #모델 등록 (best 모델 기록하기)
        mlflow.keras.log_model(model4, "model4",  registered_model_name="stock_prediction")
    #autolog 종료
        mlflow.keras.autolog(disable = True)