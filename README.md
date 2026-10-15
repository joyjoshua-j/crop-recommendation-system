# Crop Recommendation System (SDG 2: Zero Hunger)
Decision-support tool: uses all seven entered soil and weather values to predict and rank the top 3 crops with the trained model's estimated probabilities. Results explain the matching crop-profile conditions and give personalized nutrient, pH, temperature, humidity, irrigation, and drainage suggestions. Scores are model estimates, not guaranteed outcomes; this is not a replacement for an agriculture officer.

## Run (laptop)
```
pip install -r requirements.txt
# 1. Put the Kaggle file at data/Crop_recommendation.csv (columns N,P,K,temperature,humidity,ph,rainfall,label)
python train_model.py      # compares 7 models, tunes the best, saves models/ and figures/
python app.py              # open http://localhost:5000
python test_smoke.py       # optional end-to-end check
```
**Important:** without the Kaggle CSV, `train_model.py` builds a SYNTHETIC dataset so the app still runs. Its accuracy numbers are not real. Add the real file and retrain.

## Google Colab
`!pip install -r requirements.txt`, run `!python train_model.py`, then run the app in a cell with `from pyngrok import ngrok` (or `flask-ngrok`) to expose port 5000.

## Accounts
Farmers can use the app without logging in; registering saves their history. Officer login: `officer` / `officer123` (**change it** in `database.py` before real use). Set `SECRET_KEY` and `SENSOR_KEY` environment variables.

## ESP32 API
`POST /api/sensor` with header `X-API-Key: <SENSOR_KEY>` and JSON `{"soil_moisture": 41.5, "temperature": 29.3}`. The latest temperature pre-fills the form. `GET /api/sensor/latest` returns the last reading.

## Files
`train_model.py` training and figures | `predict.py` validation, top 3, advice, chart | `app.py` Flask routes | `database.py` SQLite | `templates/`, `static/` UI | `REPORT.md` method.
The PDF download is English only (default PDF fonts lack Tamil); the web UI has a Tamil toggle.
