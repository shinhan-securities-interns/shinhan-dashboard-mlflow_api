# app.py
import mlflow
from fastapi import FastAPI
from schemas import PredictIn, PredictOut

from data import preprocess_data 

import numpy as np
from databases import Database

import FinanceDataReader as fdr

import json
import os
import logging
from database.RedisDriver import RedisDriver

app = FastAPI()

DATABASE_URL = "mysql+pymysql://admin:abcd1234!@database-1.coibefbchrij.ap-northeast-2.rds.amazonaws.com:3306/mys2d"
database = Database(DATABASE_URL)

@app.on_event("startup")
async def startup_event():
    # 로그 파일 경로 및 로그 레벨 설정
    logging.basicConfig(
        filename='app.log',  # 로그 파일 경로
        level=logging.DEBUG  # 원하는 로그 레벨 설정 (예: ERROR, INFO, DEBUG 등)
    )

    print("startup_event")
    await database.connect()

    try:
        app.state.mlflow = RedisDriver(f"localhost:6322/12")

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


# REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
# REDIS_PORT = os.environ.get('REDIS_PORT', '6322')


async def predict_KOSPI() :
    stock_code = "KS11"
    key = "KOSPI_PREDICTION"
    train_data, test_data, y_test = preprocess_data(stock_code=stock_code, window_size=50, batch_size=32)
    pred = MODEL.predict(test_data)
    #print(pred)


###########나스닥 계산##################
    IXIC = fdr.DataReader('IXIC', '2013')
    IXIC_close = np.asarray(IXIC['Close'])
    IXIC_actual_prices = IXIC_close[-1] - IXIC_close[-2]
########################################

    # 2차원 배열을 1차원 리스트로 변환
    stock_predict = pred.flatten().tolist()

    # 실제 값과 예측값을 numpy 배열로 변환
    actual_prices = np.asarray(y_test)[50:]
    predicted_prices = np.asarray(pred).flatten()
    
    #예측된 값에 나스닥 영향 주기 
    predicted_changes = (predicted_prices[-1] + (2**-10 * IXIC_actual_prices)) - predicted_prices[-2]
    print("KOSPI 예측 변화율: ", predicted_changes)

    # 변화 방향 예측
    predicted_directions = "상승" if predicted_changes > 0 else "하강"
    
    result = predicted_directions
    print(predicted_directions)

    # 결과를 캐시에 저장
    await app.state.mlflow.setKey(key, result, 60 * 60 * 24)


async def predict_KOSDAQ() :
    stock_code = "KQ11"
    key = "KOSDAQ_PREDICTION"
    train_data, test_data, y_test = preprocess_data(stock_code=stock_code, window_size=50, batch_size=32)
    pred = MODEL.predict(test_data)
    #print(pred)

###########나스닥 계산##################
    IXIC = fdr.DataReader('IXIC', '2013')
    IXIC_close = np.asarray(IXIC['Close'])
    IXIC_actual_prices = IXIC_close[-1] - IXIC_close[-2]
########################################

    # 2차원 배열을 1차원 리스트로 변환
    stock_predict = pred.flatten().tolist()

    # 실제 값과 예측값을 numpy 배열로 변환
    actual_prices = np.asarray(y_test)[50:]
    predicted_prices = np.asarray(pred).flatten() 

    #예측된 값에 나스닥 영향 주기 
    predicted_changes = (predicted_prices[-1] + (2**-10 * IXIC_actual_prices)) - predicted_prices[-2]
    print("KOSDAQ 예측 변화: ", predicted_changes)


    # 변화 방향 예측
    predicted_directions = "상승" if predicted_changes > 0 else "하강"
    # predicted_directions 배열의 마지막 요소 추출
    # most_recent_prediction = predicted_directions

    result = predicted_directions
    print(predicted_directions)

    # 결과를 캐시에 저장
    await app.state.mlflow.setKey(key, result, 60 * 60 * 24)


@app.post("/predict/kospi_kosdaq")
async def get_kospi_kosdaq_prediction():

    await predict_KOSPI()
    await predict_KOSDAQ()

    return {
        "KOSPI_PREDICTION" : await app.state.mlflow.getKey("KOSPI_PREDICTION"),
        "KOSDAQ_PREDICTION": await app.state.mlflow.getKey("KOSDAQ_PREDICTION"),
    }

@app.post("/insert-prediction/")
async def insert_prediction():
    query = "INSERT INTO prediction (kospi, kosdaq) VALUES (:kospi, :kosdaq)"
    values = {"kospi": f"{await predict_KOSPI()}", "kosdaq": f"{await predict_KOSDAQ()}"}

    await database.execute(query, values)
    return {"message": "Data inserted successfully"}

