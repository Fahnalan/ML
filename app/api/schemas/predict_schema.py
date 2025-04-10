from typing import List, Optional

from pydantic import BaseModel


class StockToPredictRequestSchema(BaseModel):
    stock_ticker: str
    close: list[float]
    volumes: Optional[list[int]] = []
    model_id: int
    model_path: str
    scaler_path: str


class PredictRequestSchema(BaseModel):
    stocks: List[StockToPredictRequestSchema]


class StockFromPredictResponseSchema(BaseModel):
    stock_ticker: str
    predicted_prices: List[float]


class PredictResponseSchema(BaseModel):
    predictions: List[StockFromPredictResponseSchema]
