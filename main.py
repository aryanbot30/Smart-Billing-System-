import requests
import time
import sqlite3
import uuid
import pickle
import qrcode
import numpy as np
import RPi.GPIO as GPIO
import threading
from hx711 import HX711
from picamera2 import Picamera2
from PIL import Image
import os
import json

# ── Settings ──────────────────────────────────────────────────
CALIB_FACTOR   = 507.1
DT_PIN         = 20
SCK_PIN        = 21
MODEL_PATH        = '/home/ryan/billing/model/sklearn_model.pkl'
LEGACY_MODEL_PATH = '/home/ryan/billing/model/simple_model.npy'
LABELS_PATH       = '/home/ryan/billing/model/labels.txt'
DB_PATH        = '/home/ryan/billing/products.db'
BILLS_DIR      = '/home/ryan/billing/bills/'
TRAIN_IMG_DIR  = '/home/ryan/billing/training_images/'
PI_IP          = 'ryan.tail0fe53d.ts.net'
MIN_WEIGHT     = 30
REMOVE_BELOW   = 40

# Minimum confidence — detections below this are ignored
MIN_CONFIDENCE = 0.40

# Labels that mean "no item" — excluded from best-match selection
# Do NOT put real product names here (e.g. "nothing cmf 1" is a product)
BACKGROUND_LABELS = {'background', 'empty'}

CAPTURE_REQUEST = '/home/ryan/billing/capture_request.json'
CAPTURE_RESULT  = '/home/ryan/billing/capture_result.json'

# ── Model loader — supports sklearn .pkl and legacy centroid .npy ────────────
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                clf = pickle.load(f)
            if hasattr(clf, 'label_names_'):
                labels = clf.label_names_
            elif os.path.exists(LABELS_PATH):
                with open(LABELS_PATH) as f:
                    labels = [line.strip() for line in f if line.strip()]
            else:
                labels = []
            mtime = os.path.getmtime(MODEL_PATH)
            print(f"  [Model] Loaded sklearn RandomForest — labels: {labels}")
            return ('sklearn', clf, labels, mtime)
        except Exception as e:
            print(f"  [Model] sklearn load error: {e} — trying centroid fallback")

    if os.path.exists(LEGACY_MODEL_PATH) and os.path.exists(LABELS_PATH):
        try:
            centroids = np.load(LEGACY_MODEL_PATH)
            with open(LABELS_PATH) as f:
                labels = [line.strip() for line in f if line.strip()]
            mtime = os.path.getmtime(LEGACY_MODEL_PATH)
            print(f"  [Model] Loaded centroid model — labels: {labels}")
            return ('centroid', centroids, labels, mtime)
        except Exception as e:
            print(f"  [Model] centroid load error: {e}")

    print("  [Model] No model found — detection disabled until trained.")
    return (None, None, [], 0)

print("Loading model...")
_model_type, model, LABELS, model_mtime = load_model()

# ── GPIO + Load Cell ───────────────────────────────────────────
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
hx = HX711(dout_pin=DT_PIN, pd_sck_pin=SCK_PIN)
hx.reset()
print("Getting zero offset... keep scale empty")
time.sleep(1)
zero_readings = hx.get_raw_data(times=5)
offset = sum(zero_readings) / len(zero_readings)
print(f"Scale ready — offset: {offset:.1f}")

# ── Camera ─────────────────────────────────────────────────────
print("Starting camera...")
cam = Picamera2()
cam.start()
time.sleep(2)
print("Camera ready")

# ── Helpers ────────────────────────────────────────────────────
def get_weight():
    try:
        readings = hx.get_raw_data(times=5)
        if readings:
            avg    = sum(readings) / len(readings)
            weight = (avg - offset) / CALIB_FACTOR
            return max(0, round(weight, 1))
    except Exception as e:
        print(f"Weight error: {e}")
    return 0

def get_stable_weight(samples=8, interval=0.2):
    """Read weight `samples` times and return the median to avoid spikes."""
    readings = []
    for _ in range(samples):
        readings.append(get_weight())
        time.sleep(interval)
    readings.sort()
    mid = len(readings) // 2
    if len(readings) % 2 == 0:
        return round((readings[mid - 1] + readings[mid]) / 2, 1)
    return readings[mid]

def capture_frame_array():
    frame = cam.capture_array()
    return Image.fromarray(frame).convert('RGB')

def detect_object():
    global _model_type, model, LABELS
    if model is None or len(LABELS) == 0:
        print("  No model loaded — skipping detection")
        return None, 0
    try:
        img = capture_frame_array().resize((64, 64))
        arr = np.array(img).flatten() / 255.0

        if _model_type == 'sklearn':
            proba = model.predict_proba(arr.reshape(1, -1))[0]
            label_to_idx = {l: i for i, l in enumerate(LABELS)}
            idx_to_label = {i: l for l, i in label_to_idx.items()}
            scores = {idx_to_label.get(cls_idx, str(cls_idx)): prob
                      for cls_idx, prob in zip(model.classes_, proba)}
        else:
            centroids = model
            dists = np.linalg.norm(centroids - arr, axis=1)
            max_d = np.max(dists) if np.max(dists) > 0 else 1.0
            raw = 1 - dists / max_d
            scores = dict(zip(LABELS, raw))

        print(f"  Scores: { {k: round(v, 2) for k, v in scores.items()} }")

        best_label, best_score = None, -1
        for lbl, sc in scores.items():
            if lbl.lower() in BACKGROUND_LABELS:
                continue
            if sc > best_score:
                best_score = sc
                best_label = lbl

        if best_label is None:
            return None, 0

        # Enforce minimum confidence threshold
        if best_score < MIN_CONFIDENCE:
            print(f"  Best match: {best_label} ({best_score:.0%}) — below threshold ({MIN_CONFIDENCE:.0%}), ignoring")
            return None, best_score

        print(f"  Best match: {best_label} ({best_score:.0%})")
        return best_label, float(best_score)

    except Exception as e:
        print(f"Detection error: {e}")
    return None, 0

def get_price(name, weight_g):
    try:
        conn = sqlite3.connect(DB_PATH)
        c    = conn.cursor()
        c.execute(
            "SELECT price_per_kg, fixed_price, item_type FROM products WHERE LOWER(name)=?",
            (name.lower(),)
        )
        row = c.fetchone()
        conn.close()
        if not row:
            print(f"  Item '{name}' not found in database!")
            return 0
        if row[2] == 'weight':
            return round((weight_g / 1000) * row[0], 2)
        return row[1]
    except Exception as e:
        print(f"DB error: {e}")
        return 0

def generate_qr(bill_id, item, weight, price):
    try:
        url      = f"http://{PI_IP}:5000/bill/{bill_id}/{item}/{weight}/{price}"
        qr_image = qrcode.make(url)
        path     = f"{BILLS_DIR}{bill_id}.png"
        qr_image.save(path)
        return path, url
    except Exception as e:
        print(f"QR error: {e}")
        return None, None

def push_to_website(name, weight, price):
    try:
        requests.post('http://127.0.0.1:5000/api/add_item',
                      json={'name': name, 'weight': weight, 'price': price},
                      timeout=2)
        print(f"  Pushed to website: {name}")
    except Exception as e:
        print(f"  Website push failed: {e}")

def wait_for_removal():
    print("Remove item from scale...")
    # Always pause 2s first so loop doesn't restart instantly on 0g reads
    sleep_with_capture_check(2.0)
    # Require 6 consecutive sub-threshold readings spaced 0.3s apart
    removed = 0
    while removed < 6:
        handle_capture_request()
        time.sleep(0.3)
        w = get_weight()
        print(f"  Current weight: {w}g", end='\r')
        if w < REMOVE_BELOW:
            removed += 1
        else:
            removed = 0
    print("\nScale cleared — ready for next item\n")

# ── Model auto-reload ──────────────────────────────────────────
_model_lock = threading.Lock()

def maybe_reload_model():
    global _model_type, model, LABELS, model_mtime
    try:
        check_path = MODEL_PATH if os.path.exists(MODEL_PATH) else LEGACY_MODEL_PATH
        if not os.path.exists(check_path):
            return
        new_mtime = os.path.getmtime(check_path)
        if new_mtime > model_mtime:
            print("\n[Model] Model file changed — reloading...")
            with _model_lock:
                _model_type, model, LABELS, model_mtime = load_model()
            print(f"[Model] Now detecting: {LABELS}\n")
    except Exception as e:
        print(f"[Model] Reload check error: {e}")

def _model_watcher():
    """Background thread — checks for model file changes every 3 seconds."""
    while True:
        time.sleep(3)
        maybe_reload_model()

# Start watcher thread — reloads model within 3s of retraining
threading.Thread(target=_model_watcher, daemon=True).start()
print("[Model] Auto-reload watcher started (checks every 3s)\n")

# ── Capture request handler ────────────────────────────────────
def handle_capture_request():
    if not os.path.exists(CAPTURE_REQUEST):
        return
    try:
        with open(CAPTURE_REQUEST) as f:
            req = json.load(f)
        os.remove(CAPTURE_REQUEST)
        label     = req.get('label', 'background').lower().strip()
        req_id    = req.get('id', uuid.uuid4().hex)
        label_dir = os.path.join(TRAIN_IMG_DIR, label)
        os.makedirs(label_dir, exist_ok=True)
        fname = f"{req_id}.jpg"
        path  = os.path.join(label_dir, fname)
        img = capture_frame_array()
        img.save(path, 'JPEG', quality=90)
        print(f"[Capture] Saved training image: {path} (label={label})")
        conn = sqlite3.connect(DB_PATH)
        from datetime import datetime
        conn.execute(
            "INSERT INTO training_images (label, filepath, uploaded_at) VALUES (?,?,?)",
            (label, path, datetime.now().strftime('%Y-%m-%d %H:%M'))
        )
        conn.commit()
        conn.close()
        with open(CAPTURE_RESULT, 'w') as f:
            json.dump({'ok': True, 'path': path, 'label': label, 'id': req_id}, f)
    except Exception as e:
        print(f"[Capture] Error: {e}")
        try:
            with open(CAPTURE_RESULT, 'w') as f:
                json.dump({'ok': False, 'error': str(e)}, f)
        except Exception:
            pass

def sleep_with_capture_check(seconds):
    """Sleep for `seconds` but handle any capture request every 0.15s."""
    end = time.time() + seconds
    while time.time() < end:
        handle_capture_request()
        time.sleep(min(0.15, max(0, end - time.time())))

# ── Main Loop ──────────────────────────────────────────────────
print("System ready — place item on scale\n")

try:
    while True:
        handle_capture_request()

        weight = get_weight()
        if weight > MIN_WEIGHT:
            print(f"Item detected — weight: {weight}g")

            # Take stable median weight to filter out spikes
            weight = get_stable_weight(samples=8, interval=0.2)
            print(f"Stable weight: {weight}g — detecting object...")

            # Abort if stable weight dropped (was a noise spike)
            if weight < MIN_WEIGHT:
                print(f"Weight dropped to {weight}g — likely noise spike, ignoring\n")
                sleep_with_capture_check(0.9)
                continue

            with _model_lock:
                item, confidence = detect_object()

            if item:
                price     = get_price(item, weight)
                bill_id   = str(uuid.uuid4())[:8].upper()
                path, url = generate_qr(bill_id, item, weight, price)
                push_to_website(item, weight, price)
                print(f"\n{'='*42}")
                print(f"  Item       : {item.capitalize()}")
                print(f"  Confidence : {confidence:.0%}")
                print(f"  Weight     : {weight} g")
                print(f"  Price      : Rs. {price}")
                print(f"  Bill ID    : {bill_id}")
                print(f"  QR saved   : {path}")
                print(f"  Scan URL   : {url}")
                print(f"{'='*42}\n")
            else:
                print("Object not recognized or confidence too low — try again\n")

            wait_for_removal()

        sleep_with_capture_check(0.9)

except KeyboardInterrupt:
    print("\nShutting down...")
    GPIO.cleanup()
    cam.stop()
