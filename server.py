from flask import Flask, render_template, render_template_string, request, redirect, url_for, session, jsonify, send_file
import sqlite3, uuid, os, json, io, threading, time, subprocess, shutil
from datetime import datetime
from functools import wraps

app = Flask(__name__)
app.secret_key = 'smartbilling2024secretkey'

DB_PATH        = '/home/ryan/billing/products.db'
BILLS_DIR      = '/home/ryan/billing/bills/'
SHARED_CART    = '/home/ryan/billing/shared_cart.json'
PI_IP          = 'ryan.tail0fe53d.ts.net'
UPI_ID         = 'aryanxyz2299-2@okaxis'
UPI_NAME       = 'SmartBill+Store'
MODEL_PATH        = '/home/ryan/billing/model/sklearn_model.pkl'
LEGACY_MODEL_PATH = '/home/ryan/billing/model/simple_model.npy'
LABELS_PATH       = '/home/ryan/billing/model/labels.txt'
TRAIN_IMG_DIR  = '/home/ryan/billing/training_images/'
CAPTURE_REQUEST = '/home/ryan/billing/capture_request.json'
CAPTURE_RESULT  = '/home/ryan/billing/capture_result.json'

# retrain runs in a background thread so it never blocks Flask
_retrain_lock   = threading.Lock()
_retrain_status = {'running': False, 'msg': '', 'ok': None}

# ── Helpers ───────────────────────────────────────────────────

def load_shared_cart():
    try:
        if os.path.exists(SHARED_CART):
            with open(SHARED_CART) as f:
                return json.load(f)
    except Exception:
        pass
    return []

def save_shared_cart(items):
    try:
        with open(SHARED_CART, 'w') as f:
            json.dump(items, f)
    except Exception as e:
        print(f"Cart save error: {e}")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY, name TEXT UNIQUE,
        price_per_kg REAL, fixed_price REAL, item_type TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS bills (
        id TEXT PRIMARY KEY, items TEXT, total REAL,
        status TEXT DEFAULT 'pending', created_at TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY, username TEXT UNIQUE,
        password TEXT, role TEXT DEFAULT 'user')''')
    c.execute('''CREATE TABLE IF NOT EXISTS training_images (
        id INTEGER PRIMARY KEY, label TEXT NOT NULL,
        filepath TEXT NOT NULL, uploaded_at TEXT)''')
    c.execute("INSERT OR IGNORE INTO users (username,password,role) VALUES ('admin','admin123','admin')")
    c.execute("INSERT OR IGNORE INTO users (username,password,role) VALUES ('user1','user123','user')")
    for row in [('Onion',20.0,None,'weight'),('Potato',30.0,None,'weight'),
                ('Chips',None,20.0,'fixed'),('Kurkure',None,10.0,'fixed')]:
        c.execute("INSERT OR IGNORE INTO products (name,price_per_kg,fixed_price,item_type) VALUES (?,?,?,?)", row)
    conn.commit(); conn.close()
    os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(BILLS_DIR, exist_ok=True)

def login_required(f):
    @wraps(f)
    def d(*a, **k):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*a, **k)
    return d

def admin_required(f):
    @wraps(f)
    def d(*a, **k):
        if 'user' not in session or session.get('role') != 'admin':
            return redirect(url_for('login'))
        return f(*a, **k)
    return d

def make_upi_url(amount, bill_id):
    return f"upi://pay?pa={UPI_ID}&pn={UPI_NAME}&am={amount}&cu=INR&tn=Bill+{bill_id}"

# ── Camera singleton ──────────────────────────────────────────
_cam = None
_cam_lock = threading.Lock()

def get_cam():
    global _cam
    with _cam_lock:
        if _cam is None:
            try:
                from picamera2 import Picamera2
                _cam = Picamera2()
                _cam.start()
                time.sleep(1.5)
            except Exception as e:
                print(f"Camera init error: {e}")
    return _cam

def record_training_image(label, path):
    if not path or not os.path.exists(path):
        return
    conn = get_db()
    exists = conn.execute(
        "SELECT id FROM training_images WHERE filepath=?",
        (path,)
    ).fetchone()
    if not exists:
        conn.execute(
            "INSERT INTO training_images (label,filepath,uploaded_at) VALUES (?,?,?)",
            (label, path, datetime.now().strftime('%Y-%m-%d %H:%M'))
        )
        conn.commit()
    conn.close()

def capture_with_command(label, rotation=0):
    cmd_path = shutil.which('rpicam-still') or shutil.which('libcamera-still') or shutil.which('raspistill')
    if not cmd_path:
        return {
            'ok': False,
            'error': 'No camera command found. Install/use rpicam-still or libcamera-still, or keep main.py running.'
        }

    label_dir = os.path.join(TRAIN_IMG_DIR, label)
    os.makedirs(label_dir, exist_ok=True)
    fname = uuid.uuid4().hex + '.jpg'
    path = os.path.join(label_dir, fname)

    if os.path.basename(cmd_path) == 'raspistill':
        cmd = [cmd_path, '-n', '-t', '1000', '-o', path]
    else:
        cmd = [cmd_path, '--nopreview', '--timeout', '1000', '--output', path]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        if proc.returncode != 0 or not os.path.exists(path):
            err = (proc.stderr or proc.stdout or 'Camera command failed').strip()
            return {'ok': False, 'error': err}

        if rotation:
            try:
                from PIL import Image as PILImage
                img = PILImage.open(path).convert('RGB')
                img = img.rotate(-rotation, expand=True)
                img.save(path, 'JPEG', quality=92)
            except Exception as e:
                return {'ok': False, 'error': f'Photo captured but rotation failed: {e}'}

        record_training_image(label, path)
        return {'ok': True, 'label': label, 'file': fname, 'filepath': path, 'source': 'camera-command'}
    except subprocess.TimeoutExpired:
        return {'ok': False, 'error': 'Camera command timed out. Another program may be using the camera.'}
    except Exception as e:
        return {'ok': False, 'error': str(e)}

# ── Non-blocking capture state (req_id → result dict) ──────────
_cap_state = {}
_cap_lock  = threading.Lock()

def _bg_wait_for_capture(req_id, label, timeout=15):
    """Background thread: watches for CAPTURE_RESULT and updates _cap_state."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(0.2)
        if os.path.exists(CAPTURE_RESULT):
            try:
                with open(CAPTURE_RESULT) as f:
                    result = json.load(f)
                try:
                    os.remove(CAPTURE_RESULT)
                except Exception:
                    pass
                if result.get('ok'):
                    saved_path = result.get('filepath') or result.get('path')
                    if not saved_path and result.get('file'):
                        saved_path = os.path.join(TRAIN_IMG_DIR, label, result.get('file'))
                    try:
                        record_training_image(result.get('label', label), saved_path)
                    except Exception:
                        pass
                with _cap_lock:
                    _cap_state[req_id] = dict(result, done=True)
                return
            except Exception as e:
                with _cap_lock:
                    _cap_state[req_id] = {'ok': False, 'error': str(e), 'done': True}
                return
    # Timed out
    try:
        if os.path.exists(CAPTURE_REQUEST):
            os.remove(CAPTURE_REQUEST)
    except Exception:
        pass
    with _cap_lock:
        _cap_state[req_id] = {
            'ok': False,
            'error': 'main.py did not respond — is it running?',
            'done': True
        }

def start_capture_async(label, rotation=0):
    """Write CAPTURE_REQUEST and immediately return req_id. Result arrives via poll."""
    req_id = uuid.uuid4().hex
    try:
        if os.path.exists(CAPTURE_RESULT):
            os.remove(CAPTURE_RESULT)
    except Exception:
        pass
    try:
        with open(CAPTURE_REQUEST, 'w') as f:
            json.dump({'id': req_id, 'label': label, 'rotation': rotation}, f)
    except Exception as e:
        return None, f'Could not write capture request: {e}'

    with _cap_lock:
        _cap_state[req_id] = {'done': False}
    t = threading.Thread(target=_bg_wait_for_capture, args=(req_id, label), daemon=True)
    t.start()
    return req_id, None

# ── Background retrain (sklearn RandomForest with centroid fallback) ──────────

def _do_retrain():
    global _retrain_status
    with _retrain_lock:
        _retrain_status = {'running': True, 'msg': 'Loading images…', 'ok': None}
    try:
        import numpy as np
        from PIL import Image as PILImage

        # Check if sklearn is available
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            import pickle
            use_sklearn = True
        except ImportError:
            use_sklearn = False

        conn = get_db()
        rows = conn.execute(
            "SELECT label, filepath FROM training_images ORDER BY label"
        ).fetchall()
        conn.close()

        with _retrain_lock:
            _retrain_status['msg'] = f'Processing {len(rows)} images…'

        X = []
        y = []
        labels_found = set()

        for r in rows:
            if not os.path.exists(r['filepath']):
                continue
            try:
                arr = np.array(
                    PILImage.open(r['filepath']).convert('RGB').resize((64, 64))
                ).flatten() / 255.0
                X.append(arr)
                y.append(r['label'])
                labels_found.add(r['label'])
            except Exception as e:
                print(f"  Skip {r['filepath']}: {e}")

        if not X:
            with _retrain_lock:
                _retrain_status = {'running': False, 'msg': 'No valid images found. Add training images first.', 'ok': False}
            return

        labels = sorted(labels_found)
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

        if use_sklearn:
            # ── RandomForest (preferred) ──────────────────────────────────────
            with _retrain_lock:
                _retrain_status['msg'] = f'Training RandomForest on {len(X)} images, {len(labels)} classes…'

            label_to_idx = {l: i for i, l in enumerate(labels)}
            y_int = np.array([label_to_idx[lbl] for lbl in y])
            X_arr = np.array(X)

            min_class_count = min(sum(1 for lbl in y if lbl == l) for l in labels)

            if len(X_arr) >= 4 and min_class_count >= 2:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_arr, y_int, test_size=0.2, random_state=42, stratify=y_int
                )
                clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                clf.fit(X_train, y_train)
                acc = accuracy_score(y_test, clf.predict(X_test))
                acc_msg = f" | accuracy {acc:.0%}"
            else:
                clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                clf.fit(X_arr, y_int)
                acc_msg = " | trained on all samples"

            clf.label_names_ = labels
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(clf, f)

            method = f"RandomForest{acc_msg}"
        else:
            # ── Centroid fallback (no sklearn needed) ─────────────────────────
            with _retrain_lock:
                _retrain_status['msg'] = f'Training centroid model on {len(X)} images… (install scikit-learn for better accuracy)'

            label_images = {}
            for arr, lbl in zip(X, y):
                label_images.setdefault(lbl, []).append(arr)

            centroids = np.array([np.mean(label_images[l], axis=0) for l in labels])
            np.save(LEGACY_MODEL_PATH, centroids)
            # Touch MODEL_PATH so main.py model_status check works
            with open(MODEL_PATH + '.labels_only', 'w') as f:
                f.write('\n'.join(labels))

            method = "Centroid (install scikit-learn for better accuracy)"

        with open(LABELS_PATH, 'w') as f:
            f.write('\n'.join(labels))

        counts = {l: int(sum(1 for lbl in y if lbl == l)) for l in labels}
        msg = f"Trained {len(labels)} labels: {counts} | {method}"
        with _retrain_lock:
            _retrain_status = {'running': False, 'msg': msg, 'ok': True}

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[Retrain error]\n{tb}")
        with _retrain_lock:
            _retrain_status = {'running': False, 'msg': str(e), 'ok': False}

def start_retrain():
    t = threading.Thread(target=_do_retrain, daemon=True)
    t.start()

# ── Shared CSS ────────────────────────────────────────────────

S = """<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
:root{--bg:#0d0f12;--surface:#161920;--surface2:#1e2229;--border:#2a2f3a;
      --accent:#00e5a0;--accent2:#0066ff;--warn:#ff6b35;--text:#e8eaf0;--muted:#6b7280;
      --fh:'Syne',sans-serif;--fb:'DM Sans',sans-serif;}
*{margin:0;padding:0;box-sizing:border-box;}
body{background:var(--bg);color:var(--text);font-family:var(--fb);min-height:100vh;}
.nav{background:var(--surface);border-bottom:1px solid var(--border);padding:14px 28px;display:flex;align-items:center;justify-content:space-between;}
.brand{font-family:var(--fh);font-size:20px;font-weight:800;background:linear-gradient(90deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.nav a{color:var(--muted);text-decoration:none;font-size:14px;margin-left:20px;transition:color .2s;}
.nav a:hover{color:var(--text);}
.lo{background:var(--surface2);border:1px solid var(--border);color:var(--muted);padding:6px 14px;border-radius:6px;font-size:13px;text-decoration:none;transition:all .2s;}
.lo:hover{border-color:var(--warn)!important;color:var(--warn)!important;}
.con{max-width:1200px;margin:0 auto;padding:28px;}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:24px;}
.btn{padding:10px 20px;border-radius:8px;border:none;cursor:pointer;font-family:var(--fb);font-size:14px;font-weight:500;transition:all .2s;display:inline-flex;align-items:center;gap:6px;text-decoration:none;}
.bp{background:var(--accent);color:#000;}.bp:hover{background:#00cc8f;}
.bd{background:transparent;border:1px solid var(--warn);color:var(--warn);}.bd:hover{background:var(--warn);color:#fff;}
.bs{background:var(--surface2);border:1px solid var(--border);color:var(--text);}.bs:hover{border-color:var(--accent);color:var(--accent);}
input,select{background:var(--surface2);border:1px solid var(--border);color:var(--text);padding:10px 14px;border-radius:8px;font-family:var(--fb);font-size:14px;width:100%;outline:none;transition:border .2s;}
input[type=file]{padding:8px 10px;}
input:focus,select:focus{border-color:var(--accent);}
label{font-size:13px;color:var(--muted);margin-bottom:6px;display:block;}
h1,h2,h3{font-family:var(--fh);}
.bg{padding:3px 10px;border-radius:20px;font-size:12px;font-weight:500;}
.bgg{background:#00e5a020;color:var(--accent);border:1px solid #00e5a040;}
.bgb{background:#0066ff20;color:#4d9fff;border:1px solid #0066ff40;}
.bgw{background:#ff6b3520;color:var(--warn);border:1px solid #ff6b3540;}
.fl{padding:12px 18px;border-radius:8px;margin-bottom:20px;font-size:14px;}
.fls{background:#00e5a015;border:1px solid #00e5a030;color:var(--accent);}
.fle{background:#ff6b3515;border:1px solid #ff6b3530;color:var(--warn);}
table{width:100%;border-collapse:collapse;}
th{text-align:left;padding:12px 16px;font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);border-bottom:1px solid var(--border);font-weight:500;}
td{padding:14px 16px;border-bottom:1px solid var(--border);font-size:14px;}
tr:last-child td{border-bottom:none;}
tr:hover td{background:var(--surface2);}
.sc{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:20px;display:flex;align-items:center;gap:16px;}
.sv{font-family:var(--fh);font-size:28px;font-weight:800;color:var(--accent);}
.tabs{display:flex;gap:4px;margin-bottom:24px;background:var(--surface);padding:4px;border-radius:10px;width:fit-content;flex-wrap:wrap;}
.tab{padding:8px 20px;border-radius:7px;font-size:14px;cursor:pointer;color:var(--muted);border:none;background:transparent;font-family:var(--fb);transition:all .2s;text-decoration:none;display:inline-flex;align-items:center;}
.tab.active{background:var(--surface2);color:var(--text);border:1px solid var(--border);}
.tc{display:none;}.tc.active{display:block;}
/* Photo grid */
.pgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(100px,1fr));gap:8px;margin-top:12px;}
.pthumb{position:relative;border-radius:8px;overflow:hidden;border:2px solid var(--border);background:var(--surface2);aspect-ratio:1;}
.pthumb img{width:100%;height:100%;object-fit:cover;display:block;}
.pthumb .pdel{position:absolute;top:3px;right:3px;background:#c62828cc;color:#fff;border:none;border-radius:4px;width:20px;height:20px;font-size:11px;cursor:pointer;opacity:0;transition:opacity .2s;display:flex;align-items:center;justify-content:center;}
.pthumb:hover .pdel{opacity:1;}
.pthumb.corrupt{border-color:var(--warn);}
.pthumb.corrupt::after{content:'⚠';position:absolute;top:3px;left:4px;font-size:13px;}
.pthumb .pnum{position:absolute;bottom:3px;left:4px;font-size:9px;color:rgba(255,255,255,.6);}
/* Camera */
#cambox{width:100%;border-radius:10px;border:1px solid var(--border);background:#000;max-height:280px;object-fit:cover;}
.rot-btn{padding:6px 12px;border-radius:7px;border:1px solid var(--border);background:var(--surface2);color:var(--text);cursor:pointer;font-size:16px;transition:all .2s;}
.rot-btn:hover{border-color:var(--accent);color:var(--accent);}
.pbar-wrap{background:var(--surface2);border-radius:4px;height:6px;overflow:hidden;margin-top:6px;}
.pbar{height:100%;background:var(--accent);border-radius:4px;transition:width .3s;}
</style>"""

# ══════════════════════════════════════════════════════════════
#  Auth
# ══════════════════════════════════════════════════════════════

@app.route('/login', methods=['GET','POST'])
def login():
    err = None
    if request.method == 'POST':
        u, p = request.form['username'], request.form['password']
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE username=? AND password=?",(u,p)).fetchone()
        conn.close()
        if user:
            session['user'] = user['username']
            session['role'] = user['role']
            return redirect(url_for('admin') if user['role']=='admin' else url_for('index'))
        err = 'Invalid username or password'
    t = S + """<title>SmartBill</title>
    <style>.w{min-height:100vh;display:flex;align-items:center;justify-content:center;}</style>
    <div class="w"><div style="width:100%;max-width:400px" class="card">
    <h1 style="font-size:32px;font-weight:800;margin-bottom:6px;background:linear-gradient(90deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent">SmartBill</h1>
    <p style="color:var(--muted);margin-bottom:28px">AI-powered billing system</p>
    {% if e %}<div class="fl fle">{{e}}</div>{% endif %}
    <form method="POST">
    <div style="margin-bottom:16px"><label>Username</label><input name="username" placeholder="Enter username" required></div>
    <div style="margin-bottom:22px"><label>Password</label><input name="password" type="password" placeholder="Enter password" required></div>
    <button class="btn bp" style="width:100%;padding:12px;justify-content:center" type="submit">Sign In</button>
    </form>
    <div style="text-align:center;margin-top:16px;font-size:12px;color:var(--muted)">admin/admin123 · user1/user123</div>
    </div></div>"""
    return render_template_string(t, e=err)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ══════════════════════════════════════════════════════════════
#  User cart view
# ══════════════════════════════════════════════════════════════

@app.route('/')
@login_required
def index():
    items = load_shared_cart()
    total = round(sum(i.get('price',0) for i in items), 2)
    sid   = session.get('cart_id', str(uuid.uuid4())[:8].upper())
    session['cart_id'] = sid

    qr_url = None
    if items:
        try:
            import qrcode, base64
            qr  = qrcode.make(make_upi_url(total, sid))
            buf = io.BytesIO(); qr.save(buf, format='PNG')
            qr_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception as e:
            print(f"QR error: {e}")

    msg = request.args.get('msg')
    t = S + """<title>SmartBill — Billing</title>
    <style>
    .grid{display:grid;grid-template-columns:300px 1fr;gap:24px;align-items:start;}
    .qp{position:sticky;top:24px;}
    .qb{background:var(--surface2);border:1px solid var(--border);border-radius:12px;padding:20px;text-align:center;}
    .qi{width:200px;height:200px;background:#fff;border-radius:12px;margin:0 auto 10px;display:flex;align-items:center;justify-content:center;overflow:hidden;padding:6px;}
    .qi img{width:100%;height:100%;}
    .ta{font-family:var(--fh);font-size:34px;font-weight:800;color:var(--accent);}
    .ir{display:flex;align-items:center;gap:12px;padding:14px 0;border-bottom:1px solid var(--border);}
    .qc{display:flex;align-items:center;gap:6px;}
    .qb2{width:26px;height:26px;border-radius:6px;border:1px solid var(--border);background:var(--surface2);color:var(--text);cursor:pointer;font-size:16px;display:flex;align-items:center;justify-content:center;transition:all .2s;}
    .qb2:hover{border-color:var(--accent);color:var(--accent);}
    .upi-apps{display:flex;justify-content:center;gap:6px;margin-top:8px;flex-wrap:wrap;}
    .upi-app{font-size:11px;padding:2px 8px;border-radius:4px;background:var(--surface);border:1px solid var(--border);color:var(--muted);}
    </style>
    <div class="nav"><div class="brand">⚡ SmartBill</div>
    <div><span style="color:var(--muted);font-size:14px">👤 {{user}}</span>
    <a href="/bills" style="margin-left:20px">My Bills</a>
    <a href="/logout" class="lo" style="margin-left:16px">Logout</a></div></div>
    <div class="con">
    {% if msg %}<div class="fl fls">{{msg}}</div>{% endif %}
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px">
    <h2 style="font-size:22px">Current Session</h2>
    <button class="btn bs" onclick="clearCart()">🗑 Clear All</button></div>
    <div class="grid">
    <div class="qp">
      <div class="qb">
        <div style="font-size:12px;color:var(--muted);margin-bottom:8px;text-transform:uppercase;letter-spacing:.05em">
          {% if qr_url %}Scan & Pay via UPI{% else %}Live Bill QR{% endif %}
        </div>
        <div class="qi">
          {% if qr_url %}<img src="{{qr_url}}" alt="UPI QR">
          {% else %}<div style="color:#999;font-size:13px;text-align:center">Add items to<br>generate QR</div>{% endif %}
        </div>
        {% if qr_url %}
        <div style="background:linear-gradient(135deg,#00e5a020,#0066ff15);border:1px solid #00e5a040;border-radius:8px;padding:8px;margin-bottom:8px">
          <div style="font-size:12px;color:var(--accent);font-weight:500">Pay ₹{{total}} directly to your bank</div>
          <div style="font-size:11px;color:var(--muted);margin-top:2px">{{upi_id}}</div>
        </div>
        <div class="upi-apps">
          <span class="upi-app">GPay</span><span class="upi-app">PhonePe</span>
          <span class="upi-app">Paytm</span><span class="upi-app">BHIM</span>
        </div>
        {% else %}
        <div style="font-size:12px;color:var(--muted)">Add items to generate payment QR</div>
        {% endif %}
      </div>
      <div style="background:linear-gradient(135deg,#00e5a015,#0066ff10);border:1px solid #00e5a030;border-radius:12px;padding:18px;margin-top:14px;text-align:center">
        <div style="font-size:13px;color:var(--muted);margin-bottom:4px">Total Amount</div>
        <div class="ta" id="cart-total">₹ {{total}}</div>
        <div style="font-size:12px;color:var(--muted);margin-top:4px" id="cart-count">{{items|length}} item(s)</div>
      </div>
      {% if items %}
      <form method="POST" action="/done">
      <input type="hidden" name="session_id" value="{{sid}}">
      <button type="submit" class="btn bp" style="width:100%;padding:13px;margin-top:14px;justify-content:center;font-size:15px;font-family:var(--fh);font-weight:700">✓ Shopping Done</button></form>
      {% endif %}
    </div>
    <div class="card" id="cart-items">
    {% if items %}
      {% for item in items %}
      <div class="ir">
        <div style="flex:1">
          <div style="font-weight:500">{{item.name.capitalize()}}</div>
          <div style="font-size:13px;color:var(--muted)">
            {% if item.item_type=='weight' %}{{item.weight}}g · ₹{{item.unit_price}}/kg
            {% else %}Fixed price pack{% endif %}
          </div>
        </div>
        <div class="qc">
          <button class="qb2" onclick="updateQty({{loop.index0}},-1)">−</button>
          <span id="qty-{{loop.index0}}" style="min-width:22px;text-align:center;font-weight:500">{{item.qty}}</span>
          <button class="qb2" onclick="updateQty({{loop.index0}},1)">+</button>
        </div>
        <div id="price-{{loop.index0}}" style="font-family:var(--fh);font-weight:700;color:var(--accent);min-width:76px;text-align:right">₹ {{item.price}}</div>
        <button class="btn bd" style="padding:5px 10px;font-size:12px" onclick="removeItem({{loop.index0}})">Remove</button>
      </div>
      {% endfor %}
    {% else %}
      <div style="text-align:center;padding:48px;color:var(--muted)">
        <div style="font-size:48px;margin-bottom:12px">🛒</div>
        <div>No items yet</div>
        <div style="font-size:13px;margin-top:8px">Items appear here when main.py detects them on the scale</div>
      </div>
    {% endif %}
    </div>
    </div>
    </div>
    <script>
    var items = {{items_json|safe}};

    function removeItem(idx){
      fetch('/remove_item',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({idx:idx})}).then(()=>location.reload());
    }

    function clearCart(){
      if(!confirm('Clear all items?')) return;
      fetch('/clear_cart',{method:'POST'}).then(()=>location.reload());
    }

    function updateQty(idx, delta){
      if(idx < 0 || idx >= items.length) return;
      var item = items[idx];
      var newQty = Math.max(1, (item.qty||1) + delta);
      item.qty = newQty;
      item.price = Math.round(item.base_price * newQty * 100) / 100;
      document.getElementById('qty-'+idx).textContent = newQty;
      document.getElementById('price-'+idx).textContent = '₹ ' + item.price;
      var total = items.reduce(function(s,i){ return s + (i.price||0); }, 0);
      document.getElementById('cart-total').textContent = '₹ ' + Math.round(total*100)/100;
      document.getElementById('cart-count').textContent = items.length + ' item(s)';
      fetch('/update_cart',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({items:items})});
    }

    var lastCount = {{items|length}};
    function pollCart(){
      fetch('/api/cart_count').then(r=>r.json()).then(d=>{
        if(d.count !== lastCount){ lastCount=d.count; location.reload(); }
      }).catch(()=>{});
    }
    setInterval(pollCart, 3000);
    </script>"""
    return render_template_string(t, items=items, total=total, qr_url=qr_url,
                                  sid=sid, items_json=json.dumps(items),
                                  msg=msg, upi_id=UPI_ID,
                                  user=session.get('user'))

@app.route('/api/cart_count')
@login_required
def api_cart_count():
    return jsonify({'count': len(load_shared_cart())})

@app.route('/remove_item', methods=['POST'])
@login_required
def remove_item():
    data = request.get_json(); idx = data.get('idx', -1)
    items = load_shared_cart()
    if 0 <= idx < len(items):
        items.pop(idx); save_shared_cart(items)
    return jsonify({'ok': True})

@app.route('/clear_cart', methods=['POST'])
@login_required
def clear_cart():
    save_shared_cart([])
    session['cart_id'] = str(uuid.uuid4())[:8].upper()
    return jsonify({'ok': True})

@app.route('/update_cart', methods=['POST'])
@login_required
def update_cart():
    save_shared_cart(request.get_json().get('items', []))
    return jsonify({'ok': True})

@app.route('/done', methods=['POST'])
@login_required
def done():
    sid   = request.form.get('session_id')
    items = load_shared_cart()
    if not items:
        return redirect(url_for('index'))
    total = round(sum(i.get('price',0) for i in items), 2)
    conn  = get_db()
    conn.execute("INSERT OR REPLACE INTO bills (id,items,total,status,created_at) VALUES (?,?,?,?,?)",
                 (sid, json.dumps(items), total, 'completed',
                  datetime.now().strftime('%Y-%m-%d %H:%M')))
    conn.commit(); conn.close()
    save_shared_cart([])
    session['cart_id'] = str(uuid.uuid4())[:8].upper()
    return redirect(url_for('bills', msg='Bill completed!'))

# ── Bills ─────────────────────────────────────────────────────

@app.route('/bills')
@login_required
def bills():
    conn = get_db()
    rows = conn.execute("SELECT * FROM bills ORDER BY created_at DESC").fetchall()
    conn.close()
    bl = [{'id':r['id'],'item_count':len(json.loads(r['items'])) if r['items'] else 0,
           'total':r['total'],'status':r['status'],'created_at':r['created_at']} for r in rows]
    msg = request.args.get('msg')
    t = S + """<title>My Bills</title>
    <div class="nav"><div class="brand">⚡ SmartBill</div>
    <div><a href="/">Billing</a><a href="/logout" class="lo" style="margin-left:16px">Logout</a></div></div>
    <div class="con">
    {% if msg %}<div class="fl fls">{{msg}}</div>{% endif %}
    <h2 style="margin-bottom:20px;font-size:22px">My Bills</h2>
    {% if bills %}<div class="card"><table>
    <tr><th>Bill ID</th><th>Items</th><th>Total</th><th>Status</th><th>Date</th><th></th></tr>
    {% for b in bills %}<tr>
    <td><code style="color:var(--accent);font-size:13px">{{b.id}}</code></td>
    <td style="color:var(--muted)">{{b.item_count}} item(s)</td>
    <td><strong style="color:var(--accent)">₹ {{b.total}}</strong></td>
    <td><span class="bg bgg">{{b.status}}</span></td>
    <td style="color:var(--muted);font-size:13px">{{b.created_at}}</td>
    <td><a href="/bill_detail/{{b.id}}" class="btn bs" style="padding:5px 12px;font-size:13px">View</a></td>
    </tr>{% endfor %}</table></div>
    {% else %}<div class="card" style="text-align:center;padding:48px;color:var(--muted)">No bills yet</div>{% endif %}
    </div>"""
    return render_template_string(t, bills=bl, msg=msg)

@app.route('/bill_detail/<bill_id>')
def bill_detail(bill_id):
    conn  = get_db()
    bill  = conn.execute("SELECT * FROM bills WHERE id=?",(bill_id,)).fetchone()
    conn.close()
    if bill:
        items = json.loads(bill['items']) if bill['items'] else []
        b = {'id':bill['id'],'total':bill['total'],'status':bill['status'],'created_at':bill['created_at']}
    else:
        items = load_shared_cart()
        b = {'id':bill_id,'total':round(sum(i.get('price',0) for i in items),2),
             'status':'pending','created_at':datetime.now().strftime('%Y-%m-%d %H:%M')}
    t = S + """<title>Bill #{{b.id}}</title>
    <style>
    .bc{max-width:500px;margin:40px auto;}
    .br{display:flex;justify-content:space-between;padding:12px 0;border-bottom:1px solid var(--border);font-size:15px;}
    .bt{display:flex;justify-content:space-between;padding:16px 0;font-family:var(--fh);font-size:20px;font-weight:700;color:var(--accent);}
    .upi-apps{display:flex;justify-content:center;gap:8px;margin-top:10px;flex-wrap:wrap;}
    .upi-app{font-size:12px;padding:3px 8px;border-radius:6px;background:var(--surface2);border:1px solid var(--border);color:var(--muted);}
    </style>
    <div class="nav"><div class="brand">⚡ SmartBill</div><div><a href="/bills">← My Bills</a></div></div>
    <div class="con"><div class="bc card">
    <div style="text-align:center;margin-bottom:24px">
    <div style="font-family:var(--fh);font-size:24px;font-weight:800;background:linear-gradient(90deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent">SmartBill</div>
    <div style="color:var(--muted);font-size:13px">Bill #{{b.id}} · {{b.created_at}}</div>
    </div>
    {% for item in items %}
    <div class="br"><span>{{item.name.capitalize()}}{% if item.qty>1 %} × {{item.qty}}{% endif %}</span><span>₹ {{item.price}}</span></div>
    {% endfor %}
    <div class="bt"><span>Total</span><span>₹ {{b.total}}</span></div>
    <div style="background:linear-gradient(135deg,#00e5a015,#0066ff10);border:1px solid #00e5a030;border-radius:12px;padding:20px;text-align:center;margin-top:16px">
      <div style="font-size:14px;font-weight:500;color:var(--accent);margin-bottom:10px">Scan to Pay via UPI</div>
      <img src="/upi_qr/{{b.id}}/{{b.total}}" style="width:200px;height:200px;background:#fff;border-radius:10px;padding:8px">
      <div style="font-size:13px;margin-top:10px">₹ <strong>{{b.total}}</strong> → <span style="color:var(--accent)">{{upi_id}}</span></div>
      <div class="upi-apps">
        <span class="upi-app">GPay</span><span class="upi-app">PhonePe</span>
        <span class="upi-app">Paytm</span><span class="upi-app">BHIM</span>
      </div>
    </div>
    <div style="text-align:center;margin-top:14px"><span class="bg bgg">{{b.status}}</span></div>
    </div></div>"""
    return render_template_string(t, b=b, items=items, upi_id=UPI_ID)

@app.route('/upi_qr/<bill_id>/<total>')
def upi_qr(bill_id, total):
    import qrcode
    img = qrcode.make(make_upi_url(total, bill_id))
    buf = io.BytesIO(); img.save(buf, format='PNG'); buf.seek(0)
    return send_file(buf, mimetype='image/png')

# ── Cart API (called by main.py) ──────────────────────────────

@app.route('/api/add_item', methods=['POST'])
def api_add_item():
    data   = request.get_json()
    name   = data.get('name','').lower()
    weight = data.get('weight', 0)
    price  = data.get('price', 0)
    conn   = get_db()
    row    = conn.execute("SELECT * FROM products WHERE LOWER(name)=?",(name,)).fetchone()
    conn.close()
    if not row:
        return jsonify({'ok':False,'error':f'Product {name} not found'})
    item = {
        'name':       name,
        'weight':     weight,
        'qty':        1,
        'item_type':  row['item_type'],
        'unit_price': float(row['price_per_kg'] or row['fixed_price'] or 0),
        'base_price': float(price),
        'price':      float(price),
    }
    items = load_shared_cart(); items.append(item); save_shared_cart(items)
    print(f"API: {name} {weight}g ₹{price} — cart {len(items)} items")
    return jsonify({'ok':True,'total_items':len(items)})

# ══════════════════════════════════════════════════════════════
#  Admin dashboard
# ══════════════════════════════════════════════════════════════

@app.route('/admin')
@admin_required
def admin():
    conn     = get_db()
    bills    = conn.execute("SELECT * FROM bills ORDER BY created_at DESC").fetchall()
    products = conn.execute("SELECT * FROM products ORDER BY name").fetchall()
    tc_rows  = conn.execute(
        "SELECT label, COUNT(*) as cnt FROM training_images GROUP BY label ORDER BY label"
    ).fetchall()
    conn.close()

    train_counts = {r['label']: r['cnt'] for r in tc_rows}
    total_train  = sum(train_counts.values())

    model_exists = (os.path.exists(MODEL_PATH) or os.path.exists(LEGACY_MODEL_PATH)) and os.path.exists(LABELS_PATH)
    model_labels = []
    if os.path.exists(LABELS_PATH):
        try:
            with open(LABELS_PATH) as f:
                model_labels = [l.strip() for l in f if l.strip()]
        except Exception:
            pass

    stats = {
        'total_bills':    len(bills),
        'total_revenue':  round(sum(b['total'] or 0 for b in bills), 2),
        'total_products': len(products),
        'completed':      sum(1 for b in bills if b['status']=='completed'),
        'total_train':    total_train,
        'model_exists':   model_exists,
        'model_labels':   model_labels,
    }
    bl = [{'id':r['id'],
           'item_count':len(json.loads(r['items'])) if r['items'] else 0,
           'total':r['total'],'status':r['status'],'created_at':r['created_at']}
          for r in bills]

    all_labels = ['background'] + [p['name'].lower() for p in products]
    msg = request.args.get('msg','')
    err = request.args.get('err','')
    active_tab = request.args.get('tab', 'bills')
    if active_tab not in ('bills', 'products', 'add', 'train'):
        active_tab = 'bills'

    return render_template('admin.html', bills=bl, products=products, s=stats,
                           msg=msg, err=err, upi_id=UPI_ID,
                           train_counts=train_counts, all_labels=all_labels,
                           active_tab=active_tab)


# ── Admin: product CRUD ───────────────────────────────────────

@app.route('/admin/update_price', methods=['POST'])
@admin_required
def update_price():
    pid, it, np2 = request.form['product_id'], request.form['item_type'], float(request.form['new_price'])
    conn = get_db()
    if it == 'weight': conn.execute("UPDATE products SET price_per_kg=? WHERE id=?",(np2,pid))
    else:              conn.execute("UPDATE products SET fixed_price=? WHERE id=?",(np2,pid))
    conn.commit(); conn.close()
    return redirect(url_for('admin', tab='products', msg='Price updated!'))

@app.route('/admin/add_product', methods=['POST'])
@admin_required
def add_product():
    name = request.form['name'].strip().capitalize()
    it   = request.form['item_type']
    pk   = request.form.get('price_per_kg') or None
    fp   = request.form.get('fixed_price')  or None
    conn = get_db()
    conn.execute(
        "INSERT OR IGNORE INTO products (name,price_per_kg,fixed_price,item_type) VALUES (?,?,?,?)",
        (name, float(pk) if pk else None, float(fp) if fp else None, it)
    )
    conn.commit(); conn.close()
    os.makedirs(os.path.join(TRAIN_IMG_DIR, name.lower()), exist_ok=True)
    return redirect(url_for('admin', tab='train', msg=f'{name} added! Capture photos in Train AI tab, then click "Train Model Now". main.py will auto-reload the new model.'))

@app.route('/admin/delete_product', methods=['POST'])
@admin_required
def delete_product():
    pid = request.form['product_id']
    conn = get_db()
    conn.execute("DELETE FROM products WHERE id=?", (pid,))
    conn.commit(); conn.close()
    return redirect(url_for('admin', tab='products', msg='Product deleted.'))

# ── Admin: Training images ────────────────────────────────────

@app.route('/admin/upload_training', methods=['POST'])
@admin_required
def upload_training():
    label = request.form.get('label','').strip().lower()
    files = request.files.getlist('images')
    if not label:
        return redirect(url_for('admin', tab='train', err='Label required.'))
    if not files or files[0].filename == '':
        return redirect(url_for('admin', tab='train', err='No images selected.'))

    label_dir = os.path.join(TRAIN_IMG_DIR, label)
    os.makedirs(label_dir, exist_ok=True)
    saved = 0
    conn  = get_db()
    for f in files:
        if f and f.filename:
            ext = os.path.splitext(f.filename)[1].lower()
            if ext not in ('.jpg','.jpeg','.png','.bmp','.webp'):
                continue
            fname = uuid.uuid4().hex + ext
            path  = os.path.join(label_dir, fname)
            f.save(path)
            conn.execute(
                "INSERT INTO training_images (label,filepath,uploaded_at) VALUES (?,?,?)",
                (label, path, datetime.now().strftime('%Y-%m-%d %H:%M'))
            )
            saved += 1
    conn.commit(); conn.close()
    if saved == 0:
        return redirect(url_for('admin', tab='train', err='No valid image files (use JPG/PNG).'))
    return redirect(url_for('admin', tab='train', msg=f'{saved} image(s) uploaded for "{label}".'))

@app.route('/admin/delete_label_images', methods=['POST'])
@admin_required
def delete_label_images():
    label = request.form.get('label','').strip().lower()
    conn  = get_db()
    rows  = conn.execute("SELECT filepath FROM training_images WHERE label=?",(label,)).fetchall()
    conn.execute("DELETE FROM training_images WHERE label=?", (label,))
    conn.commit(); conn.close()
    for r in rows:
        try:
            if os.path.exists(r['filepath']): os.remove(r['filepath'])
        except Exception:
            pass
    return redirect(url_for('admin', tab='train', msg=f'All images for "{label}" deleted.'))

@app.route('/admin/delete_training_image', methods=['POST'])
@admin_required
def delete_training_image():
    data     = request.get_json() or {}
    label    = data.get('label','').strip().lower()
    filename = os.path.basename(data.get('filename',''))
    if not label or not filename:
        return jsonify({'ok':False,'error':'Missing label or filename'})
    base = os.path.realpath(TRAIN_IMG_DIR)
    path = os.path.realpath(os.path.join(TRAIN_IMG_DIR, label, filename))
    if not path.startswith(base):
        return jsonify({'ok':False,'error':'Invalid path'})
    conn = get_db()
    conn.execute("DELETE FROM training_images WHERE filepath=?", (path,))
    conn.commit(); conn.close()
    try:
        if os.path.exists(path): os.remove(path)
        return jsonify({'ok':True})
    except Exception as e:
        return jsonify({'ok':False,'error':str(e)})

@app.route('/admin/photos_for_label/<label>')
@admin_required
def photos_for_label(label):
    label     = label.strip().lower()
    label_dir = os.path.join(TRAIN_IMG_DIR, label)
    if not os.path.isdir(label_dir):
        return jsonify({'files':[]})
    files = sorted([f for f in os.listdir(label_dir)
                    if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp'))])
    return jsonify({'files': files})

@app.route('/admin/training_image/<label>/<filename>')
@admin_required
def serve_training_image(label, filename):
    base = os.path.realpath(TRAIN_IMG_DIR)
    path = os.path.realpath(os.path.join(TRAIN_IMG_DIR, label.lower(), filename))
    if not path.startswith(base) or not os.path.exists(path):
        return 'Not found', 404
    return send_file(path, mimetype='image/jpeg')

# ── Camera feed endpoint ──────────────────────────────────────

@app.route('/camera_feed')
@admin_required
def camera_feed():
    from PIL import Image as PILImage, ImageDraw
    img = PILImage.new('RGB', (640, 360), color=(18, 22, 31))
    draw = ImageDraw.Draw(img)
    draw.text((24, 140), 'Camera preview is controlled by main.py', fill=(232, 234, 240))
    draw.text((24, 170), 'Click Capture to request a training photo.', fill=(107, 114, 128))
    draw.text((24, 200), 'Keep main.py running on the Raspberry Pi.', fill=(107, 114, 128))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=85)
    return app.response_class(
        buf.getvalue(),
        mimetype='image/jpeg',
        headers={'Cache-Control':'no-cache,no-store,must-revalidate'}
    )

@app.route('/admin/capture_image', methods=['POST'])
@admin_required
def capture_image():
    data     = request.get_json() or {}
    label    = data.get('label','background').strip().lower()
    rotation = int(data.get('rotation', 0))
    if not label:
        return jsonify({'ok':False,'error':'Label required'})
    return jsonify(capture_training_photo(label, rotation))

@app.route('/admin/capture_start', methods=['POST'])
@admin_required
def capture_start():
    """Start a capture via main.py and return immediately with a req_id for polling."""
    data     = request.get_json() or {}
    label    = data.get('label', 'background').strip().lower()
    rotation = int(data.get('rotation', 0))
    if not label:
        return jsonify({'ok': False, 'error': 'Label required'})
    req_id, err = start_capture_async(label, rotation)
    if err:
        return jsonify({'ok': False, 'error': err})
    return jsonify({'ok': True, 'req_id': req_id, 'label': label})

@app.route('/admin/capture_poll/<req_id>')
@admin_required
def capture_poll(req_id):
    """Poll for the result of a capture started with /admin/capture_start."""
    with _cap_lock:
        state = _cap_state.get(req_id, {'done': False, 'error': 'Unknown request'})
    if not state.get('done'):
        return jsonify({'done': False})
    # Clean up old entries (keep last 20 only)
    with _cap_lock:
        keys = list(_cap_state.keys())
        for k in keys[:-20]:
            _cap_state.pop(k, None)
    return jsonify(dict(state, done=True))

@app.route('/admin/capture_image_direct', methods=['POST'])
@admin_required
def capture_image_direct():
    data     = request.get_json() or {}
    label    = data.get('label','background').strip().lower()
    rotation = int(data.get('rotation', 0))
    if not label:
        return jsonify({'ok':False,'error':'Label required'})
    try:
        from PIL import Image as PILImage
        cam = get_cam()
        if cam is None:
            return jsonify({'ok':False,'error':'Camera not available. Make sure picamera2 is installed.'})
        frame = cam.capture_array()
        img   = PILImage.fromarray(frame).convert('RGB')
        if rotation:
            img = img.rotate(-rotation, expand=True)
        label_dir = os.path.join(TRAIN_IMG_DIR, label)
        os.makedirs(label_dir, exist_ok=True)
        fname = uuid.uuid4().hex + '.jpg'
        path  = os.path.join(label_dir, fname)
        img.save(path, 'JPEG', quality=92)
        conn = get_db()
        conn.execute(
            "INSERT INTO training_images (label,filepath,uploaded_at) VALUES (?,?,?)",
            (label, path, datetime.now().strftime('%Y-%m-%d %H:%M'))
        )
        conn.commit(); conn.close()
        return jsonify({'ok':True,'label':label,'file':fname})
    except Exception as e:
        return jsonify({'ok':False,'error':str(e)})

# ── Train model (non-blocking — use /admin/train_poll to check status) ────────

@app.route('/admin/train_model', methods=['POST'])
@admin_required
def train_model():
    with _retrain_lock:
        if _retrain_status.get('running'):
            return jsonify({'ok': True, 'started': False, 'msg': 'Already training…'})
    start_retrain()
    return jsonify({'ok': True, 'started': True, 'msg': 'Training started…'})

@app.route('/admin/train_poll')
@admin_required
def train_poll():
    with _retrain_lock:
        st = dict(_retrain_status)
    exists = os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH)
    labels = []
    if exists:
        try:
            with open(LABELS_PATH) as f:
                labels = [l.strip() for l in f if l.strip()]
        except Exception:
            pass
    st['model_exists'] = exists
    st['model_labels'] = labels
    return jsonify(st)

@app.route('/admin/model_status')
@admin_required
def model_status():
    exists = os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH)
    labels = []
    if exists:
        try:
            with open(LABELS_PATH) as f:
                labels = [l.strip() for l in f if l.strip()]
        except Exception:
            pass
    conn   = get_db()
    counts = {r['label']:r['cnt'] for r in conn.execute(
        "SELECT label, COUNT(*) as cnt FROM training_images GROUP BY label").fetchall()}
    conn.close()
    return jsonify({'model_exists':exists,'labels':labels,'training_counts':counts})

# ── Bill pages ────────────────────────────────────────────────

@app.route('/bill/<bill_id>/<item>/<weight>/<price>')
def bill_page(bill_id, item, weight, price):
    t = S + """<title>Bill</title>
    <style>.w{min-height:100vh;display:flex;align-items:center;justify-content:center;}
    .r{display:flex;justify-content:space-between;padding:12px 0;border-bottom:1px solid var(--border);font-size:15px;}
    </style>
    <div class="w"><div style="max-width:380px;width:100%" class="card">
    <div style="text-align:center;margin-bottom:20px">
    <div style="font-family:var(--fh);font-size:22px;font-weight:800;background:linear-gradient(90deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent">SmartBill</div>
    <div style="color:var(--muted);font-size:13px">Bill #{{bid}}</div></div>
    <div class="r"><span>Item</span><strong>{{item.capitalize()}}</strong></div>
    <div class="r"><span>Weight</span><span>{{weight}} g</span></div>
    <div style="font-family:var(--fh);font-size:42px;font-weight:800;color:var(--accent);text-align:center;margin:16px 0">₹ {{price}}</div>
    <div style="background:linear-gradient(135deg,#00e5a015,#0066ff10);border:1px solid #00e5a030;border-radius:12px;padding:20px;text-align:center">
    <div style="font-size:14px;font-weight:500;color:var(--accent);margin-bottom:12px">Scan to Pay via UPI</div>
    <img src="/upi_qr/{{bid}}/{{price}}" style="width:190px;height:190px;background:#fff;border-radius:10px;padding:8px">
    <div style="font-size:12px;color:var(--muted);margin-top:10px">{{upi_id}}</div>
    <div style="font-size:11px;color:var(--muted);margin-top:4px">GPay · PhonePe · Paytm · BHIM</div>
    </div>
    <div style="text-align:center;margin-top:16px"><span class="bg bgg">Completed</span></div>
    </div></div>"""
    return render_template_string(t, bid=bill_id, item=item,
                                  weight=weight, price=price, upi_id=UPI_ID)

# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    init_db()
    print("SmartBill server starting...")
    print(f"Public URL : https://{PI_IP}")
    print(f"UPI ID     : {UPI_ID}")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
