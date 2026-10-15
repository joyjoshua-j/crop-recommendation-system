"""Crop Recommendation System - Flask app.   Run:  python train_model.py && python app.py"""
import csv, io, json, os
from functools import wraps
from flask import (Flask, render_template, request, redirect, url_for, session, flash, Response, jsonify, abort)
import database as db, predict

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-change-me")
SENSOR_KEY = os.environ.get("SENSOR_KEY", "esp32-demo-key")   # ESP32 sends this in the X-API-Key header
db.init_db()

# (key, min, max, step, label EN, label TA, help EN, help TA)
FIELDS = [
 ("N", 0, 140, 1, "Nitrogen (N)", "நைட்ரஜன் (N)", "Nitrogen in soil, kg/ha (0-140)", "மண்ணில் நைட்ரஜன், கி.கி/எக்ட். (0-140)"),
 ("P", 0, 145, 1, "Phosphorus (P)", "பாஸ்பரஸ் (P)", "Phosphorus in soil, kg/ha (0-145)", "மண்ணில் பாஸ்பரஸ், கி.கி/எக்ட். (0-145)"),
 ("K", 0, 205, 1, "Potassium (K)", "பொட்டாசியம் (K)", "Potassium in soil, kg/ha (0-205)", "மண்ணில் பொட்டாசியம், கி.கி/எக்ட். (0-205)"),
 ("temperature", 0, 50, 0.1, "Temperature (°C)", "வெப்பநிலை (°C)", "Average temperature, °C (0-50)", "சராசரி வெப்பநிலை, °C (0-50)"),
 ("humidity", 0, 100, 0.1, "Humidity (%)", "ஈரப்பதம் (%)", "Relative humidity, % (0-100)", "காற்றின் ஈரப்பதம், % (0-100)"),
 ("ph", 0, 14, 0.1, "Soil pH", "மண் pH", "Soil pH: below 7 acidic, above 7 alkaline (0-14)", "மண் pH: 7க்கு கீழ் அமிலம், மேல் காரம் (0-14)"),
 ("rainfall", 0, 500, 0.1, "Rainfall (mm)", "மழை (மி.மீ)", "Rainfall in mm (0-500)", "மழை அளவு மி.மீ (0-500)")]

L = {  # UI labels: key -> (English, Tamil)
 "title": ("Crop Recommendation System", "பயிர் பரிந்துரை அமைப்பு"),
 "subtitle": ("Enter your soil and weather values to get data-based crop suggestions.", "உங்கள் மண் மற்றும் வானிலை மதிப்புகளை உள்ளிட்டு பயிர் பரிந்துரைகளைப் பெறுங்கள்."),
 "predict": ("Recommend crops", "பயிர்களை பரிந்துரை"), "history": ("My history", "என் வரலாறு"),
 "login": ("Login", "உள்நுழை"), "logout": ("Logout", "வெளியேறு"), "register": ("Register", "பதிவு செய்"),
 "dashboard": ("Officer dashboard", "அலுவலர் பலகை"), "username": ("Username", "பயனர் பெயர்"), "password": ("Password", "கடவுச்சொல்"),
 "top3": ("Top 3 crops", "சிறந்த 3 பயிர்கள்"), "advice": ("What to improve", "என்ன மேம்படுத்த வேண்டும்"),
 "conditions": ("Your entered conditions", "நீங்கள் உள்ளிட்ட நிலைகள்"), "why": ("Why it fits", "ஏன் பொருந்துகிறது"),
 "score_note": ("Model-estimated probabilities, not guaranteed outcomes.", "மாதிரி கணித்த நிகழ்தகவுகள்; உறுதியான விளைவுகள் அல்ல."),
 "synthetic_note": ("This model was trained on synthetic data; validate recommendations locally.", "இந்த மாதிரி செயற்கைத் தரவுடன் பயிற்றுவிக்கப்பட்டது; உள்ளூரில் பரிந்துரைகளைச் சரிபார்க்கவும்."),
 "chart": ("Your values vs ideal", "உங்கள் மதிப்புகள் vs சிறந்தவை"), "pdf": ("Download PDF (English)", "PDF பதிவிறக்கு (ஆங்கிலம்)"),
 "disclaimer": ("Recommendations are estimates. Consult a local agriculture officer before making decisions.", "பரிந்துரைகள் மதிப்பீடுகள் மட்டுமே. முடிவெடுக்கும் முன் உள்ளூர் வேளாண்மை அலுவலரை அணுகவும்."),
 "date": ("Date", "தேதி"), "result": ("Result", "முடிவு"), "empty": ("No queries yet.", "இன்னும் கேள்விகள் இல்லை."),
 "top_crops": ("Most recommended crops", "அதிகம் பரிந்துரைக்கப்பட்ட பயிர்கள்"), "weekly": ("Queries per week", "வாரத்திற்கு கேள்விகள்"),
 "averages": ("Average inputs", "சராசரி உள்ளீடுகள்"), "export": ("Export CSV", "CSV ஏற்றுமதி"), "total": ("Total queries", "மொத்த கேள்விகள்")}

def lang(): return session.get("lang", "en")
@app.context_processor
def inject():
    i = 0 if lang() == "en" else 1
    return {"t": lambda k: L[k][i], "lang": lang(), "role": session.get("role"), "username": session.get("username")}

def login_required(role=None):
    def deco(fn):
        @wraps(fn)
        def wrap(*a, **kw):
            if "uid" not in session: return redirect(url_for("login", next=request.path))
            if role and session.get("role") != role: abort(403)
            return fn(*a, **kw)
        return wrap
    return deco

def fields():
    i = 0 if lang() == "en" else 1
    return [{"key": f[0], "lo": f[1], "hi": f[2], "step": f[3], "label": f[4 + i], "help": f[6 + i]} for f in FIELDS]

@app.route("/lang/<code>")
def set_lang(code):
    session["lang"] = "ta" if code == "ta" else "en"
    return redirect(request.referrer or url_for("index"))

@app.route("/")
def index(values=None):
    s = db.latest_sensor()
    vals = values or ({"temperature": s["temperature"]} if s and s["temperature"] is not None else {})
    return render_template("index.html", fields=fields(), values=vals, sensor=s)

@app.route("/predict", methods=["POST"])
def predict_route():
    vals, errors = predict.validate(request.form)
    if errors:
        for f, msg in errors.items(): flash(f"{f}: {msg}", "error")
        return index(values=request.form), 400
    qid = db.save_query(session.get("uid"), vals, predict.top3(vals))
    return redirect(url_for("result", qid=qid))

def _load(qid):
    q = db.get_query(qid) or abort(404)
    vals = {"N": q["n"], "P": q["p"], "K": q["k"], "temperature": q["temperature"], "humidity": q["humidity"], "ph": q["ph"], "rainfall": q["rainfall"]}
    return q, vals, json.loads(q["top_json"])

@app.route("/result/<int:qid>")
def result(qid):
    q, vals, top = _load(qid)
    explanations = {item["crop"]: predict.explanation(vals, item["crop"], lang()) for item in top}
    return render_template("result.html", q=q, vals=vals, top=top, explanations=explanations,
                           synthetic_model=predict.data_source().startswith("SYNTHETIC"),
                           advice=predict.advice(vals, top[0]["crop"], lang()))

@app.route("/chart/<int:qid>.png")
def chart(qid):
    q, vals, top = _load(qid)
    return Response(predict.compare_chart(vals, top[0]["crop"]), mimetype="image/png")

@app.route("/pdf/<int:qid>")
def pdf(qid):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    q, vals, top = _load(qid)
    buf = io.BytesIO(); c = canvas.Canvas(buf, pagesize=A4); y = 800   # PDF is English only (default fonts lack Tamil)
    def line(txt, size=10, dy=16):
        nonlocal y; c.setFont("Helvetica", size); c.drawString(50, y, txt[:100]); y -= dy
    line("Crop Recommendation Report", 16, 26); line(f"Date: {q['created']}", 10, 22)
    line("Inputs: " + ", ".join(f"{k}={v:g}" for k, v in vals.items()), 9, 22)
    for i, t in enumerate(top, 1):
        line(f"{i}. {t['crop']} - {t['confidence']}%", 12, 20)
        line("   Why: " + predict.explanation(vals, t["crop"], "en"), 8, 14)
    y -= 8; line("Advice:", 12, 18)
    for a in predict.advice(vals, top[0]["crop"], "en"): line("- " + a, 9)
    c.drawImage(ImageReader(io.BytesIO(predict.compare_chart(vals, top[0]["crop"]))), 50, y - 290, width=490, height=280)
    c.setFont("Helvetica-Oblique", 8); c.drawString(50, 40, L["disclaimer"][0]); c.save()
    return Response(buf.getvalue(), mimetype="application/pdf", headers={"Content-Disposition": f"attachment; filename=crop_report_{qid}.pdf"})

@app.route("/history")
@login_required()
def history():
    rows = [{"id": r["id"], "created": r["created"], "best": json.loads(r["top_json"])[0]} for r in db.user_queries(session["uid"])]
    return render_template("history.html", rows=rows)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = db.check_user(request.form.get("username", "").strip(), request.form.get("password", ""))
        if u:
            session.update(uid=u["id"], role=u["role"], username=u["username"])
            return redirect(request.args.get("next") or url_for("index"))
        flash("Wrong username or password.", "error")
    return render_template("login.html", mode="login")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        u, p = request.form.get("username", "").strip(), request.form.get("password", "")
        if len(u) < 3 or len(p) < 6: flash("Username needs 3+ characters and password 6+.", "error")
        elif not db.add_user(u, p): flash("That username is taken.", "error")
        else: flash("Account created. Please log in.", "ok"); return redirect(url_for("login"))
    return render_template("login.html", mode="register")

@app.route("/logout")
def logout():
    session.clear(); return redirect(url_for("index"))

@app.route("/officer")
@login_required("officer")
def officer():
    return render_template("dashboard.html", s=db.stats())

@app.route("/officer/export.csv")
@login_required("officer")
def export_csv():
    out = io.StringIO(); w = csv.writer(out)
    w.writerow(["id", "date", "N", "P", "K", "temperature", "humidity", "ph", "rainfall", "top_crops"])
    for r in db.all_queries():
        w.writerow([r["id"], r["created"], r["n"], r["p"], r["k"], r["temperature"], r["humidity"], r["ph"], r["rainfall"],
                    "; ".join(f"{t['crop']} {t['confidence']}%" for t in json.loads(r["top_json"]))])
    return Response(out.getvalue(), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=queries.csv"})

@app.route("/api/sensor", methods=["POST"])
def sensor_post():
    """ESP32: POST JSON {"soil_moisture": 41.5, "temperature": 29.3} with header X-API-Key."""
    if request.headers.get("X-API-Key") != SENSOR_KEY: return jsonify(error="invalid API key"), 401
    d = request.get_json(silent=True) or {}
    try: m, t = float(d["soil_moisture"]), float(d["temperature"])
    except (KeyError, TypeError, ValueError): return jsonify(error="send numeric soil_moisture and temperature"), 400
    if not (0 <= m <= 100 and 0 <= t <= 60): return jsonify(error="values out of range"), 400
    db.add_sensor(m, t); return jsonify(ok=True)

@app.route("/api/sensor/latest")
def sensor_latest():
    s = db.latest_sensor(); return jsonify(dict(s) if s else {})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
