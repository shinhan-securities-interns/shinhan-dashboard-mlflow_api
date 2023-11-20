from pydantic import BaseModel
from typing import List

class PredictIn(BaseModel):
    STOCK_CODE:str

class PredictOut(BaseModel):
    # stock_predict: List[float]
    result:str