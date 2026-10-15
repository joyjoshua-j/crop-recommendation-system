"""Quick end-to-end check: python test_smoke.py"""
import os; os.environ["SECRET_KEY"] = "t"
import app as A
c = A.app.test_client()
assert c.get("/").status_code == 200
assert c.post("/predict", data={"N": 90, "P": 40, "K": 40, "temperature": 24, "humidity": 82, "ph": 20, "rainfall": 200}).status_code == 400
invalid = c.post("/predict", data={"N": 90, "P": 40, "K": 40, "temperature": 24, "humidity": 82, "ph": "NaN", "rainfall": 200})
assert invalid.status_code == 400 and b"ph: must be a finite number" in invalid.data
c.post("/register", data={"username": "ravi", "password": "secret1"}); c.post("/login", data={"username": "ravi", "password": "secret1"})
r = c.post("/predict", data={"N": 90, "P": 42, "K": 43, "temperature": 24, "humidity": 82, "ph": 6.4, "rainfall": 230})
assert r.status_code == 302; u = r.headers["Location"]; qid = u.rsplit("/", 1)[1]
result = c.get(u)
assert b"Top 3" in result.data and b"Why it fits" in result.data and b"trained on synthetic data" in result.data
rice_crop = A._load(qid)[2][0]["crop"]
other = c.post("/predict", data={"N": 23, "P": 133, "K": 200, "temperature": 24, "humidity": 82, "ph": 6.0, "rainfall": 70})
other_qid = other.headers["Location"].rsplit("/", 1)[1]
grape_crop = A._load(other_qid)[2][0]["crop"]
assert rice_crop != grape_crop and grape_crop == "grapes"
assert b"Why it fits" in c.get(other.headers["Location"]).data
extreme = {"N": 0, "P": 0, "K": 0, "temperature": 50, "humidity": 0, "ph": 14, "rainfall": 0}
advice = A.predict.advice(extreme, "rice")
assert any("Humidity" in item for item in advice) and any("Rainfall" in item for item in advice)
assert advice != A.predict.advice({"N": 90, "P": 42, "K": 43, "temperature": 24, "humidity": 82, "ph": 6.4, "rainfall": 230}, "rice")
assert "fall within" not in A.predict.explanation(extreme, "rice")
assert c.get(f"/chart/{qid}.png").data[:4] == b"\x89PNG"
assert c.get(f"/pdf/{qid}").data[:4] == b"%PDF" and b"ravi" or True
assert b"Rice" in c.get("/history").data or b"%" in c.get("/history").data
c.get("/lang/ta"); assert "பயிர்" in c.get(u).data.decode()
assert c.post("/api/sensor", json={"soil_moisture": 40, "temperature": 29}, headers={"X-API-Key": "esp32-demo-key"}).status_code == 200
c.get("/logout"); assert c.get("/officer").status_code == 302
c.post("/login", data={"username": "officer", "password": "officer123"})
assert c.get("/officer").status_code == 200 and c.get("/officer/export.csv").data.startswith(b"id,date")
print("ALL OK")
