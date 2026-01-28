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
def ui():
    df = pd.read_csv("tables/2025/ht_api_input.csv")
    df['Date'] = pd.to_datetime(df.Date)
    time = datetime.now() + timedelta(hours=-8)
    df = df[df.Date == str(time.date())]
    player_list = df.sort_values('Player').Player.unique().tolist()

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NBA Predictor</title>
    <style>
        /* Reset and body styling */
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: flex-start;
            min-height: 100vh;
            background-color: #f5f5f5;
        }}

        .container {{
            width: 100%;
            max-width: 500px; /* looks good on desktop */
            background: #fff;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}

        h1 {{
            font-size: 1.8rem;
            text-align: center;
            margin-bottom: 20px;
        }}

        label {{
            font-weight: bold;
            display: block;
            margin-bottom: 8px;
        }}

        select, button {{
            width: 100%;
            padding: 12px;
            font-size: 16px;
            margin-bottom: 16px;
            border-radius: 8px;
            border: 1px solid #ccc;
            box-sizing: border-box;
        }}

        button {{
            background-color: #007bff;
            color: white;
            border: none;
            cursor: pointer;
            transition: background-color 0.2s;
        }}

        button:hover {{
            background-color: #0056b3;
        }}

        #result {{
            font-size: 1.1rem;
            text-align: center;
            margin-top: 10px;
        }}

        /* Responsive text */
        @media (max-width: 480px) {{
            h1 {{
                font-size: 1.4rem;
            }}
            select, button {{
                padding: 10px;
                font-size: 14px;
            }}
            #result {{
                font-size: 1rem;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>NBA Live Points Predictor</h1>

        <label for="player">Select player:</label>
        <select id="player">
            <option value="">--Select a player--</option>
        </select>

        <button onclick="predict()">Predict</button>

        <div id="result"></div>
    </div>

    <script>
        const players = {player_list};
        const select = document.getElementById("player");
        players.forEach(p => {{
            const opt = document.createElement("option");
            opt.value = p;
            opt.text = p;
            select.add(opt);
        }});

        async function predict() {{
            const player = select.value;
            const resultDiv = document.getElementById("result");
            if (!player) {{
                resultDiv.innerHTML = "<p style='color:red;'>Please select a player!</p>";
                return;
            }}
            resultDiv.innerHTML = "Loading...";

            try {{
                const res = await fetch("/predict", {{
                    method: "POST",
                    headers: {{ "Content-Type": "application/json" }},
                    body: JSON.stringify({{ player_name: player }})
                }});

                if (!res.ok) {{
                    const text = await res.text();
                    throw new Error(`Server error: ${{res.status}}\\n${{text}}`);
                }}

                const data = await res.json();
                resultDiv.innerHTML = `
                    <p><b>${{data.player}}</b></p>
                    <p>Current: ${{data.current_pts}} pts in ${{data.current_mins}} mins</p>
                    <p><b>Predicted Final: ${{data.predicted_final_pts}} pts</b></p>
                `;
            }} catch (err) {{
                resultDiv.innerHTML = `<p style="color:red;">Error: ${{err.message}}</p>`;
                console.error(err);
            }}
        }}
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)

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
        df[col] = df[col].replace('--', 0).astype(int)
    
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
    ht_booster_mean = xgb.Booster()
    ht_booster_mean.load_model("ML_models/ht_PTS_mean_model.json")
    ht_model_mean = XGBRegressor()
    ht_model_mean._Booster = ht_booster_mean

    ht_booster_Qlow = xgb.Booster()
    ht_booster_Qlow.load_model("ML_models/ht_PTS_Qlow_model.json")
    ht_model_Qlow = XGBRegressor()
    ht_model_Qlow._Booster = ht_booster_Qlow

    ht_booster_Qhigh = xgb.Booster()
    ht_booster_Qhigh.load_model("ML_models/ht_PTS_Qhigh_model.json")
    ht_model_Qhigh = XGBRegressor()
    ht_model_Qhigh._Booster = ht_booster_Qhigh
    
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
        for col in ['PTS', 'FG', 'FGA']:
            df.loc[df['Player'] == req.player_name, f'{col}Diff'] = df[f'{col}_h1'] - df[f'{col}_h1_base']
        df = df.drop(['MP'], axis=1)
        pts_prediction_qlow = int(round(ht_model_Qlow.predict(df)[0], 0))
        pts_prediction_mean = int(round(ht_model_mean.predict(df)[0], 0))
        pts_prediction_qhigh = int(round(ht_model_Qhigh.predict(df)[0], 0))
        
        spread_factor = (df['Spread_h1'].abs() / 20).clip(lower=0, upper=1).iloc[0]
        if df['Spread_h1'].iloc[0] < 0:
            # favored → ceiling matters more
            pts_prediction = int(round(pts_prediction_mean * (1 - spread_factor) +
                                pts_prediction_qhigh * spread_factor, 0))
        else:
            # underdog → floor matters more
            pts_prediction = int(round(pts_prediction_mean * (1 - spread_factor) +
                                pts_prediction_qlow * spread_factor, 0))

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