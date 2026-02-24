from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from datetime import datetime, timedelta
import os
import requests
from io import StringIO
import json
import pandas as pd

from xgboost import XGBRegressor
import xgboost as xgb

YEAR = 2025

# -------------------------
# Create FastAPI app
# -------------------------
app = FastAPI(title="Test NBA API", version="0.1")

# -------------------------
# Pydantic models
# -------------------------
class PredictionRequest(BaseModel):
    player_name: str
    mode: str

class PredictionResponse(BaseModel):
    player: str
    current_mins: int
    current_fgm: int
    current_fga: int
    current_tpm: int
    current_tpa: int
    current_ftm: int
    current_fta: int
    current_pts: int
    current_fouls: int
    team_pts: int
    opp_pts: int
    predicted_final_pts: int
    pts_prediction_qlow: int
    pts_prediction_qhigh: int
    pregame_mins_preds: float
    pregame_pts_preds: float
    pts_line: float

# -------------------------
# Routes
# -------------------------
def load_gh_artfct(url, mode):
    token = os.getenv("GITHUB_PAT")
    r = requests.get(url, headers={"Authorization": f"token {token}"})

    if mode == 'table':
        df = pd.read_csv(StringIO(r.text))
        df['Date'] = pd.to_datetime(df.Date)
        return df
    else:
        booster = xgb.Booster()
        booster.load_model(bytearray(r.content))
        return booster

def generate_section(prefix):
    return f"""
    <label>Select Team:</label>
    <select id="{prefix}-team">
        <option value="">--Select a team--</option>
    </select>

    <label>Select Player:</label>
    <select id="{prefix}-player" disabled>
        <option value="">--Select a player--</option>
    </select>

    <button onclick="sendPrediction('/predict', '{prefix}')">
        Predict
    </button>

    <div id="{prefix}-result" class="result"></div>
    """
@app.get("/", response_class=HTMLResponse)
def ui():
    df = load_gh_artfct(f"https://raw.githubusercontent.com/Relenes510/fantasy_basketball/refs/heads/main/tables/{YEAR}/ht_api_input.csv", "table")
    df['Date'] = pd.to_datetime(df.Date)

    time = datetime.now() + timedelta(hours=-8)
    df = df[df.Date == str(time.date())]

    team_players = (df[['Team', 'Player']].drop_duplicates().sort_values(['Team', 'Player']).groupby('Team')['Player'].apply(list).to_dict())
    team_players_json = json.dumps(team_players)

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NBA Predictor</title>

    <style>
    body {{
        font-family: Arial, sans-serif;
        margin: 0;
        padding: 20px;
        display: flex;
        justify-content: center;
        background-color: #f5f5f5;
    }}

    .container {{
        width: 100%;
        max-width: 520px;
        background: #fff;
        padding: 24px;
        border-radius: 14px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.1);
    }}

    h1 {{
        text-align: center;
        margin-bottom: 20px;
    }}

    .tabs {{
        display: flex;
        margin-bottom: 20px;
    }}

    .tab-btn {{
        flex: 1;
        padding: 10px;
        border: none;
        background: #e0e0e0;
        font-weight: bold;
        cursor: pointer;
    }}

    .tab-btn.active {{
        background: #007bff;
        color: white;
    }}

    .section {{
        display: none;
    }}

    .section.active {{
        display: block;
    }}

    label {{
        font-weight: bold;
        margin-bottom: 6px;
        display: block;
    }}

    select, button {{
        width: 100%;
        padding: 12px;
        font-size: 16px;
        margin-bottom: 16px;
        border-radius: 8px;
        border: 1px solid #ccc;
    }}

    button {{
        background-color: #007bff;
        color: white;
        border: none;
        cursor: pointer;
    }}

    button:hover {{
        background-color: #0056b3;
    }}

    .result {{
        text-align: center;
        font-size: 1.05rem;
    }}
    </style>
    </head>

    <body>
    <div class="container">

    <h1>NBA Points Predictor</h1>

    <div class="tabs">
        <button class="tab-btn active" onclick="switchTab('pregame')">Pregame</button>
        <button class="tab-btn" onclick="switchTab('live')">Live</button>
    </div>

    <!-- LIVE SECTION -->
    <div id="live" class="section">
        {generate_section("live")}
    </div>

    <!-- PREGAME SECTION -->
    <div id="pregame" class="section active">
        {generate_section("pregame")}
    </div>

    </div>

    <script>
    const teamPlayers = {team_players_json};

    /* -------------------------
    TAB SWITCHING
    ------------------------- */
    function switchTab(tabName) {{
        document.querySelectorAll('.section')
            .forEach(sec => sec.classList.remove('active'));

        document.querySelectorAll('.tab-btn')
            .forEach(btn => btn.classList.remove('active'));

        document.getElementById(tabName).classList.add('active');
        event.target.classList.add('active');
    }}

    /* -------------------------
    DROPDOWN INITIALIZER
    ------------------------- */
    function initializeDropdowns(prefix) {{
        const teamSelect = document.getElementById(prefix + "-team");
        const playerSelect = document.getElementById(prefix + "-player");

        Object.keys(teamPlayers).sort().forEach(team => {{
            const opt = document.createElement("option");
            opt.value = team;
            opt.textContent = team;
            teamSelect.appendChild(opt);
        }});

        teamSelect.addEventListener("change", () => {{
            playerSelect.innerHTML = '<option value="">--Select a player--</option>';
            const players = teamPlayers[teamSelect.value];

            if (!players) {{
                playerSelect.disabled = true;
                return;
            }}

            players.forEach(p => {{
                const opt = document.createElement("option");
                opt.value = p;
                opt.textContent = p;
                playerSelect.appendChild(opt);
            }});

            playerSelect.disabled = false;
        }});
    }}

    /* -------------------------
    API CALL
    ------------------------- */
    async function sendPrediction(endpoint, prefix) {{
        const player = document.getElementById(prefix + "-player").value;
        const resultDiv = document.getElementById(prefix + "-result");

        if (!player) {{
            resultDiv.innerHTML = "<p style='color:red;'>Select a player</p>";
            return;
        }}

        resultDiv.innerHTML = "Loading...";

        try {{
            const res = await fetch(endpoint, {{
                method: "POST",
                headers: {{ "Content-Type": "application/json" }},
                body: JSON.stringify({{ player_name: player, mode: prefix }})
            }});

            const data = await res.json();

            if (prefix == "live") {{

                resultDiv.innerHTML = `
                    <p><b>${{data.player}}</b></p>
                    <p>Current Stats: ${{data.current_pts}} pts in ${{data.current_mins}} mins</p>
                    <p>${{data.current_fgm}} for ${{data.current_fga}} FG | ${{data.current_tpm}} for ${{data.current_tpa}} 3PT | ${{data.current_ftm}} for ${{data.current_fta}} FT
                    <p>Player Fouls: ${{data.current_fouls}} | TeamPts: ${{data.team_pts}} | OppPts: ${{data.opp_pts}}</p>
                    <p>Pregame Predicted Stats: ${{data.pregame_pts_preds}} pts in ${{data.pregame_mins_preds}} mins</p>
                    <p><b>Predicted Final Ranges: Low: ${{data.pts_prediction_qlow}} pts / Avg: ${{data.predicted_final_pts}} pts / High: ${{data.pts_prediction_qhigh}} pts</b></p>
                `;

            }} else if (prefix == "pregame") {{

                resultDiv.innerHTML = `
                    <p><b>${{data.player}}</b></p>
                    <p><b>Projected Points:</b> ${{data.pregame_pts_preds}}</p>
                    <p><b>Projected Minutes:</b> ${{data.pregame_mins_preds}}</p>
                    <p><b>DraftKings Points Line:</b> ${{data.pts_line}}</p>
                `;
            }}

        }} catch (err) {{
            resultDiv.innerHTML = `<p style="color:red;">Player Unavailable</p>`;
        }}
    }}

    /* -------------------------
    INIT
    ------------------------- */
    initializeDropdowns("live");
    initializeDropdowns("pregame");
    </script>

    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
def health():
    df = load_gh_artfct(f"https://raw.githubusercontent.com/Relenes510/fantasy_basketball/refs/heads/main/tables/{YEAR}/ht_api_input.csv", "table")
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
            continue

        for i, team_data in enumerate(players_teams):
            team = team_data.get("team", {})
            team_abbr = team.get("abbreviation", "")
            opp_abbr = players_teams[1-i]["team"].get("abbreviation", "")
            
            stats_blocks = team_data.get("statistics", [])
            if not stats_blocks:
                continue

            stats_block = stats_blocks[0]
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
    df = df.drop(['FG', '3PT', 'FT'], axis=1, errors='ignore')
    df = df.rename(columns={"MIN": "MP", "3PTM": "TPM", "3PTA": "TPA", "FGM": "FG", "FTM": "FT", "OREB": "ORB", "TO": "TOV"})
    for col in df.columns.difference(['TEAM', 'PLAYER', 'STARTER', 'OPP']):
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    for col in ['PTS', 'FGA', 'FTA', 'ORB', 'TOV']:
        df[f'Team{col}'] = (df.sort_values(['TEAM']).groupby(['TEAM'])[col].transform('sum'))
        if col in ['PTS', 'FGA']:
            df[f'Team{col}_pct'] = df[f'{col}'] / df[f'Team{col}']
    
    df['OppTeamPTS'] = df['OPP'].map(df.groupby('TEAM')['TeamPTS'].first().to_dict())
    df['Spread'] = df['TeamPTS'] - df['OppTeamPTS']
    
    df['Player_Pace'] = df.apply(lambda row: (row['FGA'] + 0.44 * row['FTA']) / row['MP'] if row['MP'] > 0 else 0, axis=1)
    df['Team_Pace'] = ((df['TeamFGA'] + 0.44 * df['TeamFTA'] - df['TeamORB'] + df['TeamTOV']) / 120)
    df['Player_Pace_Rel'] = df['Player_Pace'] / df['Team_Pace']
    df['Pace_Minutes_Interaction'] = df['Player_Pace'] * df['MP']
    
    return df

@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    time = datetime.now() + timedelta(hours=-8)

    df_lines = load_gh_artfct(f"https://raw.githubusercontent.com/Relenes510/fantasy_basketball/refs/heads/main/tables/{YEAR}/parlay_lines.csv", "table")
    df_lines = df_lines[(df_lines.Player == req.player_name) & (df_lines.Date == str(time.date()))]
    if df_lines.shape[0] > 0:
        pts_line = float(df_lines['PTS_line'].iloc[0])
    else:
        pts_line = 0

    if req.mode == "pregame":
        df = load_gh_artfct(f"https://raw.githubusercontent.com/Relenes510/fantasy_basketball/refs/heads/main/tables/{YEAR}/ht_api_input.csv", "table")
        df = df[(df.Player == req.player_name) & (df.Date == str(time.date()))]
        return {
            "player": req.player_name,
            "current_mins": 0,
            "current_pts": 0,
            "current_fgm": 0,
            "current_fga": 0,
            "current_tpm": 0,
            "current_tpa": 0,
            "current_ftm": 0,
            "current_fta": 0,
            "current_fouls": 0,
            "team_pts": 0,
            "opp_pts": 0,
            "predicted_final_pts": 0,
            "pts_prediction_qlow": 0,
            "pts_prediction_qhigh": 0,
            "pregame_mins_preds": float(round(df['MP_proj'].iloc[0], 2)),
            "pregame_pts_preds": float(round(df['PTS_proj'].iloc[0], 2)),
            "pts_line": pts_line
        }

    ht_booster_mean = load_gh_artfct("https://raw.githubusercontent.com/Relenes510/fantasy_basketball/refs/heads/main/ML_models/ht_PTS_mean_model.json", "booster")
    ht_model_mean = XGBRegressor()
    ht_model_mean._Booster = ht_booster_mean

    ht_booster_Qlow = load_gh_artfct("https://raw.githubusercontent.com/Relenes510/fantasy_basketball/refs/heads/main/ML_models/ht_PTS_Qlow_model.json", "booster")
    ht_model_Qlow = XGBRegressor()
    ht_model_Qlow._Booster = ht_booster_Qlow

    ht_booster_Qhigh = load_gh_artfct("https://raw.githubusercontent.com/Relenes510/fantasy_basketball/refs/heads/main/ML_models/ht_PTS_Qhigh_model.json", "booster")
    ht_model_Qhigh = XGBRegressor()
    ht_model_Qhigh._Booster = ht_booster_Qhigh
    
    df = load_gh_artfct(f"https://raw.githubusercontent.com/Relenes510/fantasy_basketball/refs/heads/main/tables/{YEAR}/ht_api_input.csv", "table")
    
    df['Team'] = df['Team'].astype('category')
    df['Opp'] = df['Opp'].astype('category')
    df['Player'] = df['Player'].astype('category')
    df['Pos'] = df['Pos'].astype('category')
    df = df[(df.Date == str(time.date())) & (df.Player == req.player_name)].drop(['Season', 'Date', 'PTS'], axis=1)

    df_ht = get_live_stat()
    df_ht = df_ht[df_ht.PLAYER == req.player_name]
    team_pts = int(df_ht['TeamPTS'].iloc[0])
    opp_team_pts = int(df_ht['OppTeamPTS'].iloc[0])
    df_ht = df_ht.drop(['TeamPTS', 'OppTeamPTS', 'TeamFGA'], axis=1)
    if df_ht.shape[0] > 0:
        for catg in ['MP', 'PTS', 'FG', 'FGA', 'FT', 'FTA', 'TPM', 'TPA', 'PF', 'TeamPTS_pct', 'TeamFGA_pct', 'Spread']:
            if catg in ['TeamPTS_pct', 'TeamFGA_pct']:
                ht_stat = df_ht[catg].iloc[0]
            else:
                ht_stat = int(df_ht[catg].iloc[0])
            df.loc[df['Player'] == req.player_name, f'{catg}_h1'] = ht_stat
            
        df.loc[df['Player'] == req.player_name, 'Player_Pace_Rel'] = df_ht['Player_Pace_Rel'].iloc[0]
        df.loc[df['Player'] == req.player_name, 'Pace_Minutes_Interaction'] = df_ht['Pace_Minutes_Interaction'].iloc[0]
        for col in ['MP', 'PTS', 'FG', 'FGA']:
            df.loc[df[f'{col}_proj'] > 0, f'{col}_proj_pct'] = ((df[f'{col}_h1'] - df[f'{col}_proj']) / df[f'{col}_proj'])
        
        pts_prediction_qlow = int(round(ht_model_Qlow.predict(df)[0], 0))
        pts_prediction_mean = int(round(ht_model_mean.predict(df)[0], 0))
        pts_prediction_qhigh = int(round(ht_model_Qhigh.predict(df)[0], 0))

        return {
            "player": req.player_name,
            "current_mins": int(df_ht['MP'].iloc[0]),
            "current_pts": int(df_ht['PTS'].iloc[0]),
            "current_fgm": int(df_ht['FG'].iloc[0]),
            "current_fga": int(df_ht['FGA'].iloc[0]),
            "current_tpm": int(df_ht['TPM'].iloc[0]),
            "current_tpa": int(df_ht['TPA'].iloc[0]),
            "current_ftm": int(df_ht['FT'].iloc[0]),
            "current_fta": int(df_ht['FTA'].iloc[0]),
            "current_fouls": int(df_ht['PF'].iloc[0]),
            "team_pts": team_pts,
            "opp_pts": opp_team_pts,
            "predicted_final_pts": pts_prediction_mean,
            "pts_prediction_qlow": pts_prediction_qlow,
            "pts_prediction_qhigh": pts_prediction_qhigh,
            "pregame_mins_preds": float(round(df['MP_proj'].iloc[0], 2)),
            "pregame_pts_preds": float(round(df['PTS_proj'].iloc[0], 2)),
            "pts_line": pts_line
        }
    else:
        return {
            "player": "Player Unavailable",
            "current_mins": 0,
            "current_pts": 0,
            "current_fgm": 0,
            "current_fga": 0,
            "current_tpm": 0,
            "current_tpa": 0,
            "current_ftm": 0,
            "current_fta": 0,
            "current_fouls": 0,
            "team_pts": 0,
            "opp_pts": 0,
            "predicted_final_pts": 0,
            "pts_prediction_qlow": 0,
            "pts_prediction_qhigh": 0,
            "pregame_mins_preds": 0,
            "pregame_pts_preds": 0,
            "pts_line": 0
        }