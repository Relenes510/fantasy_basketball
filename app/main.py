from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from datetime import datetime, timedelta
import requests
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

class PredictionResponse(BaseModel):
    player: str
    current_pts: int
    current_mins: int
    predicted_final_pts: int

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
    df = pd.read_csv("tables/2025/ht_api_input.csv")
    time =  datetime.now() + timedelta(hours=-8)
    return {"status": "ok", "rows": df.shape[0], "time": time}



def get_live_stat():
    response = requests.get("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard")
    data = response.json()
    
    games = data.get('events', [])
    rows = []

    for game in games:
        game_id = game['id']
        summary = requests.get(f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}").json()
        boxscore = summary.get("boxscore", {})

        players_teams = boxscore.get("players", [])
        if len(players_teams) < 2:
            continue  # skip if something weird happens

        # Determine opponent for each team
        for i, team_data in enumerate(players_teams):
            team = team_data.get("team", {})
            team_abbr = team.get("abbreviation", "")
            opp_abbr = players_teams[1-i]["team"].get("abbreviation", "")
            
            stats_blocks = team_data.get("statistics", [])
            if not stats_blocks:
                continue

            stats_block = stats_blocks[0]  # Player-level stats
            labels = stats_block.get("labels", stats_block.get("names", []))
            athletes = stats_block.get("athletes", [])

            for p in athletes:
                stats = p.get("stats", [])
                if not stats:
                    continue

                row = dict(zip(labels, stats))
                row["PLAYER"] = p["athlete"]["displayName"]
                row["TEAM"] = team_abbr
                row["OPP"] = opp_abbr
                row["STARTER"] = p.get("starter", False)
                rows.append(row)

    df = pd.DataFrame(rows)
    for col in ['FG', '3PT', 'FT']:
        if col in df.columns:
            df[f'{col}M'] = df[col].str.split('-').str[0]
            df[f'{col}A'] = df[col].str.split('-').str[1]
    df = df.drop(['FG', '3PT', 'FT'], axis=1).rename(columns={"MIN": "MP", "3PTM": "TPM", "3PTA": "TPA", "FGM": "FG", "FTM": "FT"})
    for col in df.columns.difference(['TEAM', 'PLAYER', 'STARTER', 'OPP']):
        df[col] = df[col].astype(int)
    
    df['TeamPTS'] = (df.sort_values(['TEAM']).groupby(['TEAM'])['PTS'].transform('sum'))
    df['TeamPTS_pct'] = df['PTS'] / df['TeamPTS']
    df['TeamFGA'] = (df.sort_values(['TEAM']).groupby(['TEAM'])['FGA'].transform('sum'))
    df['TeamFGA_pct'] = df['FGA'] / df['TeamFGA']
    
    df['OppTeamPTS'] = df['OPP'].map(df.groupby('TEAM')['TeamPTS'].first().to_dict())
    df['Spread'] = df['TeamPTS'] - df['OppTeamPTS']
    
    df = df.drop(['TeamPTS', 'OppTeamPTS', 'TeamFGA'], axis=1)
    
    return df

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

    time = datetime.now() + timedelta(hours=-8)
    df = df[(df.Date == str(time.date())) & (df.Player == req.player_name)].drop(['Season', 'Date', 'PTS'], axis=1)

    df_ht = get_live_stat()
    df_ht = df_ht[df_ht.PLAYER == req.player_name]

    if df_ht.shape[0] > 0:
        for catg in ['MP', 'PTS', 'FG', 'FGA', 'FT', 'FTA', 'TPM', 'TPA', 'PF', 'TeamPTS_pct', 'TeamFGA_pct', 'Spread']:
            if catg in ['TeamPTS_pct', 'TeamFGA_pct']:
                ht_stat = df_ht[catg].iloc[0]
            else:
                ht_stat = int(df_ht[catg].iloc[0])
            df.loc[df['Player'] == req.player_name, f'{catg}_h1'] = ht_stat

        df.loc[(df['role'] == 2) & (df['MP_h1'] < 5), 'role'] = 3
        df.loc[df['Player'] == req.player_name, 'MP_h2'] = df['MP'] - df['MP_h1']
        df.loc[df['Player'] == req.player_name, 'PTSDiff'] = df['PTS_h1'] - df['PTS_h1_base']
        df = df.drop(['MP'], axis=1)
        pts_prediction = int(round(ht_model.predict(df)[0], 0))

        return {
            "player": req.player_name,
            "current_mins": int(df_ht['MP'].iloc[0]),
            "current_pts": int(df_ht['PTS'].iloc[0]),
            "predicted_final_pts": pts_prediction
        }
    else:
        return {
            "player": "Player Unavailable",
            "current_mins": 0,
            "current_pts": 0,
            "predicted_final_pts": 0
        }