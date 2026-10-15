"""Prediction helpers: validation, top-3 crops, rule-based advice, comparison chart."""
import io, json, math
from pathlib import Path
import joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).parent
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
FEATURE_LABELS = {
    "N": {"en": "nitrogen", "ta": "நைட்ரஜன்"},
    "P": {"en": "phosphorus", "ta": "பாஸ்பரஸ்"},
    "K": {"en": "potassium", "ta": "பொட்டாசியம்"},
    "temperature": {"en": "temperature", "ta": "வெப்பநிலை"},
    "humidity": {"en": "humidity", "ta": "ஈரப்பதம்"},
    "ph": {"en": "pH", "ta": "pH"},
    "rainfall": {"en": "rainfall", "ta": "மழை"},
}
# Valid input ranges (min, max) used for form validation and chart scaling
RANGES = {"N": (0, 140), "P": (0, 145), "K": (0, 205), "temperature": (0, 50),
          "humidity": (0, 100), "ph": (0, 14), "rainfall": (0, 500)}
_cache = {}

def _load():
    if not _cache:
        _cache["model"] = joblib.load(BASE / "models/crop_model.joblib")
        with open(BASE / "models/crop_profiles.json", encoding="utf-8") as f:
            _cache["profiles"] = json.load(f)
        with open(BASE / "models/metrics.json", encoding="utf-8") as f:
            _cache["metrics"] = json.load(f)
    return _cache

def validate(form):
    """Return (values, errors). Errors are keyed by feature name."""
    vals, errors = {}, {}
    for f in FEATURES:
        raw = str(form.get(f, "")).strip()
        try:
            v = float(raw)
        except ValueError:
            errors[f] = "must be a number"; continue
        if not math.isfinite(v):
            errors[f] = "must be a finite number"; continue
        lo, hi = RANGES[f]
        if not lo <= v <= hi:
            errors[f] = f"must be between {lo} and {hi}"
        vals[f] = v
    return vals, errors

def top3(vals):
    """Top 3 crops with confidence (%)."""
    m = _load()["model"]
    proba = m.predict_proba(pd.DataFrame([vals], columns=FEATURES))[0]
    idx = np.argsort(proba)[::-1][:3]
    return [{"crop": str(m.classes_[i]), "confidence": round(float(proba[i]) * 100, 1)} for i in idx]

def profile(crop):
    return _load()["profiles"][crop]

def data_source():
    return _load()["metrics"]["data_source"]

# Advice templates: key -> {language: text}
ADV = {
 "N_low": {"en": "Nitrogen is low: add urea or compost / farmyard manure.", "ta": "நைட்ரஜன் குறைவாக உள்ளது: யூரியா அல்லது தொழு உரம் / கம்போஸ்ட் இடவும்."},
 "N_high": {"en": "Nitrogen is high: skip nitrogen fertilizer this season.", "ta": "நைட்ரஜன் அதிகமாக உள்ளது: இந்த பருவத்தில் நைட்ரஜன் உரத்தைத் தவிர்க்கவும்."},
 "P_low": {"en": "Phosphorus is low: add DAP or single super phosphate.", "ta": "பாஸ்பரஸ் குறைவாக உள்ளது: DAP அல்லது சூப்பர் பாஸ்பேட் இடவும்."},
 "P_high": {"en": "Phosphorus is high: avoid phosphate fertilizer.", "ta": "பாஸ்பரஸ் அதிகமாக உள்ளது: பாஸ்பேட் உரத்தைத் தவிர்க்கவும்."},
 "K_low": {"en": "Potassium is low: add muriate of potash (MOP) or wood ash.", "ta": "பொட்டாசியம் குறைவாக உள்ளது: மியூரேட் ஆஃப் பொட்டாஷ் (MOP) இடவும்."},
 "K_high": {"en": "Potassium is high: avoid potash fertilizer.", "ta": "பொட்டாசியம் அதிகமாக உள்ளது: பொட்டாஷ் உரத்தைத் தவிர்க்கவும்."},
 "ph_low": {"en": "Soil pH ({value:.1f}) is below the typical {crop} range ({low:.1f}-{high:.1f}); retest the soil and consider agricultural lime if locally advised.", "ta": "மண் pH ({value:.1f}) {crop} பயிரின் வழக்கமான வரம்பை ({low:.1f}-{high:.1f}) விடக் குறைவு; மண்ணை மீண்டும் பரிசோதித்து உள்ளூர் ஆலோசனையின்படி சுண்ணாம்பு பயன்படுத்தவும்."},
 "ph_high": {"en": "Soil pH ({value:.1f}) is above the typical {crop} range ({low:.1f}-{high:.1f}); check soil alkalinity and use a locally recommended amendment.", "ta": "மண் pH ({value:.1f}) {crop} பயிரின் வழக்கமான வரம்பை ({low:.1f}-{high:.1f}) விட அதிகம்; மண் காரத்தன்மையைச் சரிபார்த்து உள்ளூர் பரிந்துரைக்கப்பட்ட திருத்தத்தைப் பயன்படுத்தவும்."},
 "temperature_low": {"en": "Temperature ({value:.1f} C) is below the typical {crop} range ({low:.1f}-{high:.1f} C); consider a locally adapted variety or planting window.", "ta": "வெப்பநிலை ({value:.1f} C) {crop} பயிரின் வழக்கமான வரம்பை ({low:.1f}-{high:.1f} C) விடக் குறைவு; உள்ளூருக்கு ஏற்ற வகை அல்லது நடவு காலத்தைப் பரிசீலிக்கவும்."},
 "temperature_high": {"en": "Temperature ({value:.1f} C) is above the typical {crop} range ({low:.1f}-{high:.1f} C); consider a locally adapted variety or planting window.", "ta": "வெப்பநிலை ({value:.1f} C) {crop} பயிரின் வழக்கமான வரம்பை ({low:.1f}-{high:.1f} C) விட அதிகம்; உள்ளூருக்கு ஏற்ற வகை அல்லது நடவு காலத்தைப் பரிசீலிக்கவும்."},
 "humidity_low": {"en": "Humidity ({value:.1f}%) is below the typical {crop} range ({low:.1f}-{high:.1f}%); monitor soil moisture and irrigate as needed.", "ta": "ஈரப்பதம் ({value:.1f}%) {crop} பயிரின் வழக்கமான வரம்பை ({low:.1f}-{high:.1f}%) விடக் குறைவு; மண் ஈரத்தை கண்காணித்து தேவைக்கேற்ப பாசனம் செய்யவும்."},
 "humidity_high": {"en": "Humidity ({value:.1f}%) is above the typical {crop} range ({low:.1f}-{high:.1f}%); monitor for fungal disease and maintain airflow between plants.", "ta": "ஈரப்பதம் ({value:.1f}%) {crop} பயிரின் வழக்கமான வரம்பை ({low:.1f}-{high:.1f}%) விட அதிகம்; பூஞ்சை நோயைக் கண்காணித்து செடிகளுக்கிடையே காற்றோட்டத்தைப் பராமரிக்கவும்."},
 "rain_low": {"en": "Rainfall ({value:.1f} mm) is below the typical {crop} range ({low:.1f}-{high:.1f} mm); plan supplemental irrigation, preferably drip where practical.", "ta": "மழை ({value:.1f} மி.மீ) {crop} பயிரின் வழக்கமான வரம்பை ({low:.1f}-{high:.1f} மி.மீ) விடக் குறைவு; இயன்றால் சொட்டு நீர் மூலம் கூடுதல் பாசனத்தைத் திட்டமிடவும்."},
 "rain_high": {"en": "Rainfall ({value:.1f} mm) is above the typical {crop} range ({low:.1f}-{high:.1f} mm); provide field drainage to reduce waterlogging.", "ta": "மழை ({value:.1f} மி.மீ) {crop} பயிரின் வழக்கமான வரம்பை ({low:.1f}-{high:.1f} மி.மீ) விட அதிகம்; நீர் தேங்காமல் வயல் வடிகாலை ஏற்படுத்தவும்."},
 "ok": {"en": "Your measured nutrients, pH, temperature, humidity, and rainfall fall within the typical ranges for {crop}.", "ta": "உங்கள் ஊட்டச்சத்துகள், pH, வெப்பநிலை, ஈரப்பதம் மற்றும் மழை {crop} பயிரின் வழக்கமான வரம்புகளுக்குள் உள்ளன."},
}

def advice(vals, crop, lang="en"):
    """Personalized advice compared with the crop's typical (10th-90th percentile) range."""
    p, messages = profile(crop), []
    for f in ("N", "P", "K"):
        if vals[f] < p[f]["low"]: messages.append(ADV[f + "_low"][lang])
        elif vals[f] > p[f]["high"]: messages.append(ADV[f + "_high"][lang])
    if vals["ph"] < p["ph"]["low"]:
        messages.append(ADV["ph_low"][lang].format(crop=crop, value=vals["ph"], **p["ph"]))
    elif vals["ph"] > p["ph"]["high"]:
        messages.append(ADV["ph_high"][lang].format(crop=crop, value=vals["ph"], **p["ph"]))
    for f in ("temperature", "humidity", "rainfall"):
        if vals[f] < p[f]["low"]:
            key = "rain_low" if f == "rainfall" else f + "_low"
            messages.append(ADV[key][lang].format(crop=crop, value=vals[f], **p[f]))
        elif vals[f] > p[f]["high"]:
            key = "rain_high" if f == "rainfall" else f + "_high"
            messages.append(ADV[key][lang].format(crop=crop, value=vals[f], **p[f]))
    if not messages:
        messages.append(ADV["ok"][lang].format(crop=crop))
    return messages

def explanation(vals, crop, lang="en"):
    """Explain the crop-profile conditions that most closely match these inputs."""
    p = profile(crop)
    matching = [f for f in FEATURES if p[f]["low"] <= vals[f] <= p[f]["high"]]
    in_range = bool(matching)
    if not in_range:
        def distance(f):
            if vals[f] < p[f]["low"]:
                delta = p[f]["low"] - vals[f]
            else:
                delta = vals[f] - p[f]["high"]
            return delta / (RANGES[f][1] - RANGES[f][0])
        matching = sorted(FEATURES, key=distance)[:2]
    else:
        matching = sorted(
            matching,
            key=lambda f: abs(vals[f] - p[f]["mean"]) / max(p[f]["high"] - p[f]["low"], 1e-9),
        )[:3]
    labels = ", ".join(FEATURE_LABELS[f][lang] for f in matching)
    if lang == "ta":
        if not in_range:
            return f"உங்கள் உள்ளீடுகளில் {labels} மதிப்புகள் {crop} பயிரின் வழக்கமான வரம்புகளுக்கு அருகில் உள்ளன; மற்ற அளவுகள் அந்த வரம்புகளுக்கு வெளியே உள்ளன."
        return f"{labels} மதிப்புகள் {crop} பயிரின் வழக்கமான வரம்புகளுடன் பொருந்துகின்றன; பயிர் சுயவிவர ஒப்பீடு இந்தப் பரிந்துரையை ஆதரிக்கிறது."
    if not in_range:
        return f"Your closest conditions are {labels}; these are nearest to {crop}'s typical ranges, while the other measured values fall outside them."
    return f"Your {labels} values fall within {crop}'s typical range, supporting this match in the model's crop-profile comparison."

def compare_chart(vals, crop):
    """PNG bytes: farmer's values vs ideal range of the crop (scaled to 0-100% of each valid range)."""
    p = profile(crop)
    sc = lambda f, v: (v - RANGES[f][0]) / (RANGES[f][1] - RANGES[f][0]) * 100
    x, w = np.arange(len(FEATURES)), 0.38
    you = [sc(f, vals[f]) for f in FEATURES]
    mean = np.array([sc(f, p[f]["mean"]) for f in FEATURES])
    lo = np.array([sc(f, p[f]["low"]) for f in FEATURES]); hi = np.array([sc(f, p[f]["high"]) for f in FEATURES])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, you, w, label="Your field", color="#2e7d32")
    ax.bar(x + w/2, mean, w, label=f"Ideal for {crop}", color="#ffb300")
    ax.errorbar(x + w/2, mean, yerr=[mean - lo, hi - mean], fmt="none", ecolor="black", capsize=3)
    ax.set_xticks(x); ax.set_xticklabels([f"{f}\n{vals[f]:g} vs {p[f]['mean']:.0f}" for f in FEATURES], fontsize=7)
    ax.set_ylabel("% of valid range"); ax.legend(fontsize=8); ax.set_title("Your values vs ideal (bar = mean, line = typical range)", fontsize=9)
    fig.tight_layout(); buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110); plt.close(fig)
    return buf.getvalue()
