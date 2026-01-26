# Basketball Stat Prediction Model

- Author: Rodolfo Elenes
- Email contact: rodolfoe7157@gmail.com
- Halftime Points Model Predictor
- [https://fantasy-basketball-zbmx.onrender.com/](Halftime Points Model Predictor)

Hello! This project is about creating and maintaining ML models that help assist in creating player prop bets for sportsbook betting. 
There are active CI/CD pipeline processes in place through Github Actions + cronho.st that actively update my tables with starting lineups

# Process
- Everyday at 8:30am PST I trigger my add_new_data CI/CD pipeline
  - to pickup the previous day's boxscore data
  - update NBA schedule table with possible rescheduled or new games
  - update NBA roster file
- I also run shortly after and in multiple intervals throughout the week my gen_preds CI/CD pipeline
  - to pickup sportsbook odds
  - get current starting lineups
  - generate ML predictions based off acquired data

# Project Progression
Originally I had set up this repository to store NBA boxscore data to do analysis to aid me in my fantasy basketball league.
After some days of data collection, I thought to myself that I can generate an ML model with this information. I then learned that github actions
existed where I already had some experience with CI/CD pipelines to help me get started. I was excited to find out that I can automatically refresh 
my dataset while being away from home. I had to use cronho.st because Github Actions cron scheduler has issues in scheduling on time. I then began to 
inquire about my other data sources to properly evaluate my model performances and enhance its performance. I first worked on a two step process where 
the player's minutes are predicted first, then the points. I eventually tried an ensembling approach by adding two residual points models. Residual points 
is the difference between the actual points and final sportsbook line, I am attempting to catch any significant mistake by sportsbook makers. One of these 
models is a classification model and the other is a regression model. After spending some time on experimenting  ML engineering, I thought about an idea 
of creating a model that predicts from halftime. I was getting to a point where I realized that there is limited amount of pregame context thats available 
that won't expect the randomness within sports. So I thought about how knowing half the game already eliminates most of the randomness that exists in sports. 
I ended up finding Render, a service where I can host my API. And currently that is where my work is at.

## Data
This project uses anonymized or publicly available data.
No proprietary or confidential data is included.

## Acknowledgements
This project uses the Python package nbainjuries by
[@mxufc29](https://github.com/mxufc29)

and Public ESPN API by 
[@pseudo-r](https://github.com/pseudo-r/Public-ESPN-API?tab=readme-ov-file)

Thanks to the authors for making it publicly available.
