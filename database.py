"""SQLite storage: users (hashed passwords), saved queries, ESP32 sensor readings."""
import json, sqlite3
from collections import Counter
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

DB = "crop.db"
def conn():
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row; return c

def init_db():
    with conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, username TEXT UNIQUE, password_hash TEXT, role TEXT DEFAULT 'farmer');
        CREATE TABLE IF NOT EXISTS queries(id INTEGER PRIMARY KEY, user_id INTEGER, created TEXT, n REAL, p REAL, k REAL,
            temperature REAL, humidity REAL, ph REAL, rainfall REAL, top_json TEXT);
        CREATE TABLE IF NOT EXISTS sensor(id INTEGER PRIMARY KEY, created TEXT, moisture REAL, temperature REAL);""")
        if not c.execute("SELECT 1 FROM users WHERE username='officer'").fetchone():   # default officer: CHANGE the password!
            c.execute("INSERT INTO users(username,password_hash,role) VALUES('officer',?,'officer')", (generate_password_hash("officer123"),))

def add_user(username, password):
    try:
        with conn() as c:
            c.execute("INSERT INTO users(username,password_hash) VALUES(?,?)", (username, generate_password_hash(password)))
        return True
    except sqlite3.IntegrityError:
        return False

def check_user(username, password):
    with conn() as c: u = c.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    return u if u and check_password_hash(u["password_hash"], password) else None

def save_query(user_id, v, top):
    with conn() as c:
        cur = c.execute("INSERT INTO queries(user_id,created,n,p,k,temperature,humidity,ph,rainfall,top_json) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (user_id, datetime.now().strftime("%Y-%m-%d %H:%M"), v["N"], v["P"], v["K"], v["temperature"], v["humidity"], v["ph"], v["rainfall"], json.dumps(top)))
        return cur.lastrowid

def get_query(qid):
    with conn() as c: return c.execute("SELECT * FROM queries WHERE id=?", (qid,)).fetchone()
def user_queries(uid):
    with conn() as c: return c.execute("SELECT * FROM queries WHERE user_id=? ORDER BY id DESC", (uid,)).fetchall()
def all_queries():
    with conn() as c: return c.execute("SELECT * FROM queries ORDER BY id DESC").fetchall()

def stats():
    """Officer dashboard numbers."""
    with conn() as c:
        rows = c.execute("SELECT top_json FROM queries").fetchall()
        weekly = c.execute("SELECT strftime('%Y-W%W',created) w, COUNT(*) n FROM queries GROUP BY w ORDER BY w DESC LIMIT 12").fetchall()
        avg = c.execute("SELECT COUNT(*) total, AVG(n) n, AVG(p) p, AVG(k) k, AVG(rainfall) rain FROM queries").fetchone()
    return {"top": Counter(json.loads(r[0])[0]["crop"] for r in rows).most_common(8),
            "weekly": [dict(w) for w in weekly][::-1], "avg": dict(avg)}

def add_sensor(moisture, temperature):
    with conn() as c: c.execute("INSERT INTO sensor(created,moisture,temperature) VALUES(?,?,?)", (datetime.now().strftime("%Y-%m-%d %H:%M"), moisture, temperature))
def latest_sensor():
    with conn() as c: return c.execute("SELECT * FROM sensor ORDER BY id DESC LIMIT 1").fetchone()
