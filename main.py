# app.py
import mlflow
from fastapi import FastAPI
from schemas import PredictIn, PredictOut
from data import preprocess_data 
import numpy as np


import json
import os
import logging
import database.RedisDriver

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    # 로그 파일 경로 및 로그 레벨 설정
    logging.basicConfig(
        filename='app.log',  # 로그 파일 경로
        level=logging.DEBUG  # 원하는 로그 레벨 설정 (예: ERROR, INFO, DEBUG 등)
    )

    print("startup_event")
    try:
        app.state.mlflow = database.RedisDriver.RedisDriver(f"{REDIS_HOST}:{REDIS_PORT}/12")

    except Exception as e:
        print(f"An error occurred during startup: {e}")


def get_model_production():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    model_uri = "models:/stock_prediction/production" 
    model_p = mlflow.keras.load_model(model_uri)
    return model_p
MODEL = get_model_production()

###################################3
# def get_model():
#     model = mlflow.keras.load_model(model_uri="./model")
#     return model
# MODEL = get_model()


REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = os.environ.get('REDIS_PORT', '6322')


async def predict_KOSPI() :
    stock_code = "KS11"
    key = "KOSPI_PREDICTION"
    train_data, test_data, y_test = preprocess_data(stock_code=stock_code, window_size=50, batch_size=32)
    pred = MODEL.predict(test_data)
    print(pred)

    # 2차원 배열을 1차원 리스트로 변환
    stock_predict = pred.flatten().tolist()
    print(stock_predict)


    # 캐시된 데이터가 없으면 데이터 생성 및 JSON 직렬화 후 Redis에 저장

    # 실제 값과 예측값을 numpy 배열로 변환
    actual_prices = np.asarray(y_test)[50:]
    predicted_prices = np.asarray(pred).flatten()  # pred가 2차원 배열인 경우 flatten 사용
    # 실제 값과 예측값의 다음 날 변화 계산
    actual_changes = actual_prices[1:] - actual_prices[:-1]
    predicted_changes = (predicted_prices[-1]) - predicted_prices[-2]
    # 변화 방향 예측
    predicted_directions = "up" if predicted_changes > 0 else "down"
    # predicted_directions 배열의 마지막 요소 추출
    most_recent_prediction = predicted_directions

    result = most_recent_prediction
    print(most_recent_prediction)

    # 결과를 캐시에 저장
    await app.state.mlflow.setKey(key, result, 60 * 60 * 24)

async def predict_KOSDAQ() :
    stock_code = "KQ11"
    key = "KOSDAQ_PREDICTION"
    train_data, test_data, y_test = preprocess_data(stock_code=stock_code, window_size=50, batch_size=32)
    pred = MODEL.predict(test_data)
    print(pred)

    # 2차원 배열을 1차원 리스트로 변환
    stock_predict = pred.flatten().tolist()
    print(stock_predict)


    # 캐시된 데이터가 없으면 데이터 생성 및 JSON 직렬화 후 Redis에 저장

    # 실제 값과 예측값을 numpy 배열로 변환
    actual_prices = np.asarray(y_test)[50:]
    predicted_prices = np.asarray(pred).flatten()  # pred가 2차원 배열인 경우 flatten 사용
    # 실제 값과 예측값의 다음 날 변화 계산
    actual_changes = actual_prices[1:] - actual_prices[:-1]
    predicted_changes = (predicted_prices[-1]) - predicted_prices[-2]
    # 변화 방향 예측
    predicted_directions = "up" if predicted_changes > 0 else "down"
    # predicted_directions 배열의 마지막 요소 추출
    most_recent_prediction = predicted_directions

    result = most_recent_prediction
    print(most_recent_prediction)

    # 결과를 캐시에 저장
    await app.state.mlflow.setKey(key, result, 60 * 60 * 24)

@app.get("/predict/kospi_kosdaq")
async def get_kospi_kosdaq_prediction():

    await predict_KOSDAQ()
    await predict_KOSDAQ()

    return {
        "KOSPI_PREDICTION" : await app.state.mlflow.getKey("KOSPI_PREDICTION"),
        "KOSDAQ_PREDICTION": await app.state.mlflow.getKey("KOSDAQ_PREDICTION"),
    }

