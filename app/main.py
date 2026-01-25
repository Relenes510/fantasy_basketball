from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from datetime import datetime
import pandas as pd

from xgboost import XGBRegressor
import xgboost as xgb
# -------------------------
# Create FastAPI app
# -------------------------
app = FastAPI(title="Test NBA API", version="0.1")

# -------------------------
# Pydantic models
# -------------------------
class PredictionRequest(BaseModel):
    player_name: str
    ht_pts: float

class PredictionResponse(BaseModel):
    player: str
    predicted_final_pts: float

# -------------------------
# Routes
# -------------------------
@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
    <head><title>NBA API Home</title></head>
    <body>
    <h1>Welcome to the NBA API!</h1>
    <p>Click <a href='/docs'>here</a> to explore the API.</p>
    </body>
    </html>
    """

@app.get("/health")
def health():
    return {"status": "ok", "rows": df.shape[0], "time": datetime.now()}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    ht_booster = xgb.Booster()
    ht_booster.load_model("ML_models/ht_PTS_model.json")
    ht_model = XGBRegressor()
    ht_model._Booster = ht_booster

    df = pd.read_csv("tables/2025/ht_api_input.csv")
    df['Date'] = pd.to_datetime(df.Date)
    df['Team'] = df['Team'].astype('category')
    df['Opp'] = df['Opp'].astype('category')
    df['Player'] = df['Player'].astype('category')
    df['Pos'] = df['Pos'].astype('category')
    df = df[(df.Date == str(datetime.now().date())) & (df.Player == req.player_name)].drop(['Season', 'Date', 'PTS'], axis=1)
    df.loc[df['Player'] == req.player_name, 'PTS_h1'] = req.ht_pts
    pts_prediction = int(ht_model.predict(df))

    return {
        "player": req.player_name,
        "predicted_final_pts": pts_prediction
    }