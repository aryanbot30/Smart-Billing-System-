# ⚡ Smart Billing System

> **AI-powered automated billing for small vendors — Raspberry Pi 4 · Load Cell · Camera · UPI Payment**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.x-black?logo=flask)](https://flask.palletsprojects.com)
[![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-4-red?logo=raspberrypi)](https://raspberrypi.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![SDG](https://img.shields.io/badge/UN%20SDG-1%20%7C%202%20%7C%208%20%7C%209%20%7C%2011%20%7C%2012-blue)](https://sdgs.un.org)

---

## 📌 What is it?

SmartBill is a low-cost, fully automated digital billing system designed for kirana stores, street vendors, and rural markets. It replaces expensive POS terminals (₹15,000–₹80,000) with a ₹4,200 Raspberry Pi setup.

**Place an item on the scale → camera identifies it → price calculated → UPI QR generated → bill logged.**  
No manual entry. No paper. No cash counting errors.

---

## ✨ Features

| Feature | Detail |
|---|---|
| 🤖 **AI Object Detection** | Random Forest classifier on 64×64 RGB pixel vectors |
| ⚖ **Auto Weight** | HX711 ADC + load cell with median stabilisation |
| 💳 **UPI Payment** | Auto QR for GPay, PhonePe, Paytm, BHIM |
| 📊 **Web Dashboard** | Flask admin — live cart, bills, products |
| 🔄 **Auto Model Reload** | Retrain from browser; main.py picks up in 3s |
| 📷 **In-browser Training** | Capture training photos from Pi camera via admin panel |
| 📥 **Excel Export** | Weekly/monthly bill reports + product sales analysis |
| 🌐 **Remote Access** | Tailscale tunnel — manage from anywhere |
| 🗑 **No Drift** | Adaptive weight stabilisation handles Lays/Kurkure (light items) |

---

## 🖼 Hardware Connection Diagram

![Connection Diagram](docs/connection_diagram.svg)

### Pin Mapping

#### Load Cell → HX711 (colour-coded wires)

| Load Cell Wire | HX711 Pin | Notes |
|---|---|---|
| Red | E+ | Excitation positive |
| Black | E− | Excitation negative |
| White | A+ | Signal positive |
| Green | A− | Signal negative |

#### HX711 → Raspberry Pi 4 GPIO (BCM numbering)

| HX711 Pin | Pi 4 Pin | GPIO | Wire colour |
|---|---|---|---|
| VCC | Pin 1 | 3.3 V | Red |
| GND | Pin 6 | Ground | Black |
| DT | Pin 38 | GPIO 20 | Green |
| SCK | Pin 40 | GPIO 21 | Blue |

#### Pi Camera v2

| Camera | Pi 4 |
|---|---|
| 15-pin ribbon | CSI camera port (between USB and HDMI) |

> **Calibration factor:** `507.1` — adjust in `main.py` → `CALIB_FACTOR` using a known weight.

---

## 🗂 Project Structure

```
smart-billing-system/
├── main.py               # Detection loop — load cell + camera + AI
├── server.py             # Flask web server — cart, admin, billing
├── admin.html            # Admin dashboard template
├── requirements.txt      # Python dependencies
├── setup.sh              # One-shot Pi setup script
├── .gitignore
├── LICENSE
├── docs/
│   └── connection_diagram.svg   # Hardware wiring diagram
├── model/
│   └── .gitkeep          # sklearn_model.pkl and labels.txt go here (not tracked)
└── scripts/
    └── install_service.sh        # systemd service installer
```

---

## 🚀 Quick Start

### 1 — Hardware

- Raspberry Pi 4 (2 GB or 4 GB RAM)
- HX711 load cell amplifier module
- Load cell (1 kg or 5 kg)
- Raspberry Pi Camera Module v2
- MicroSD card ≥ 32 GB (Raspberry Pi OS Bookworm 64-bit)
- 5 V / 3 A USB-C power supply

Wire according to the [connection diagram](#-hardware-connection-diagram) above.

### 2 — Software setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/smart-billing-system.git
cd smart-billing-system

# Run the one-shot setup script
chmod +x setup.sh
./setup.sh
```

Or install manually:

```bash
pip install -r requirements.txt --break-system-packages
```

### 3 — Configure paths

Edit the top of `main.py` and `server.py`:

```python
# main.py
PI_IP         = 'YOUR_TAILSCALE_OR_LOCAL_IP'   # e.g. '192.168.1.100'
CALIB_FACTOR  = 507.1        # calibrate with a known weight
MIN_WEIGHT    = 30           # grams — ignore readings below this
MIN_CONFIDENCE = 0.40        # 0–1 detection threshold
```

```python
# server.py
PI_IP   = 'YOUR_TAILSCALE_OR_LOCAL_IP'
UPI_ID  = 'your-upi-id@bank'
UPI_NAME = 'Your Store Name'
```

### 4 — Run

Open **two terminals** on the Pi:

```bash
# Terminal 1 — web server (start this first)
python3 server.py

# Terminal 2 — detection loop
python3 main.py
```

Open your browser: `http://YOUR_PI_IP:5000`

Admin dashboard: `http://YOUR_PI_IP:5000/admin`  
Login: `admin` / `admin123` (change in production!)

---

## 🤖 Training the AI Model

The model is a **Random Forest Classifier** (scikit-learn) trained on 64×64 RGB pixel feature vectors.

### Via Admin Dashboard (recommended)

1. Go to **Admin → Train AI tab**
2. Select a product label
3. Click **📸 Capture Photos** — Pi camera saves training images automatically
4. Aim for **50+ photos per product** with varied angles and lighting
5. Click **🚀 Train Model Now**
6. `main.py` auto-reloads the new model within 3 seconds — no restart needed

### Tips for good accuracy

- Include a **`background`** label (empty scale photos)
- Vary **lighting, angle, and position** within each class
- Keep items **centred** under the camera
- Avoid reflective or transparent packaging

### Model files

| File | Description |
|---|---|
| `model/sklearn_model.pkl` | Primary — Random Forest (scikit-learn) |
| `model/simple_model.npy` | Fallback — Nearest Centroid (numpy only) |
| `model/labels.txt` | Label names, one per line |

Both files are in `.gitignore` — train them on your own Pi.

---

## 📊 Admin Dashboard

| Tab | Features |
|---|---|
| **Bills** | Full bill history · ⬇ Download Weekly Excel · ⬇ Download Monthly Excel |
| **Products** | Add / Delete / Update price for each product |
| **Add Product** | Name, type (weight-based or fixed price), pricing |
| **Train AI** | Capture photos · Upload from computer · Per-label photo grid · Delete bad images · Retrain |

### Excel Export

Three report types available from the Bills tab:

- **This Week** — Bills from the last 7 days (Bills Summary + Item Details sheets)
- **This Month** — Bills from the last 30 days
- **Product Sales Report** — Product Summary + Daily Sales + Product Catalogue (3 sheets)

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `flask` | Web server |
| `picamera2` | Pi Camera capture |
| `RPi.GPIO` | GPIO control |
| `hx711` | Load cell reading |
| `numpy` | Array operations |
| `Pillow` | Image processing |
| `scikit-learn` | Random Forest classifier |
| `qrcode` | UPI QR generation |
| `requests` | HTTP between main.py and server |
| `openpyxl` | Excel report export |

Full list in `requirements.txt`.

---

## 🔧 Running as a System Service

To start both scripts automatically on boot:

```bash
chmod +x scripts/install_service.sh
sudo ./scripts/install_service.sh
```

Or manually:

```bash
# /etc/systemd/system/smartbill-server.service
sudo systemctl enable smartbill-server
sudo systemctl start smartbill-server

# /etc/systemd/system/smartbill-main.service
sudo systemctl enable smartbill-main
sudo systemctl start smartbill-main
```

---

## 🌍 SDG Alignment

This project contributes to **6 UN Sustainable Development Goals**:

| SDG | Contribution |
|---|---|
| 🔴 **SDG 1** No Poverty | ₹4,200 setup enables 63M+ micro-enterprises to digitise |
| 🟡 **SDG 2** Zero Hunger | Per-gram billing prevents over-charging for fresh produce |
| 🟤 **SDG 8** Decent Work | 240s → 30s per transaction; digital records for micro-loans |
| 🟠 **SDG 9** Innovation | IoT + AI on a single-board computer for street retail |
| 🟠 **SDG 11** Smart Cities | Digital retail infrastructure for informal markets |
| 🟡 **SDG 12** Responsible Consumption | Paperless billing; accurate weighing reduces food waste |

---

## ⚙ Configuration Reference

```python
# main.py — key constants
CALIB_FACTOR   = 507.1        # Calibration — adjust per your load cell
DT_PIN         = 20           # HX711 data pin (BCM)
SCK_PIN        = 21           # HX711 clock pin (BCM)
MIN_WEIGHT     = 30           # g — minimum weight to trigger detection
REMOVE_BELOW   = 40           # g — item considered removed below this
MIN_CONFIDENCE = 0.40         # 0.0–1.0 — reject detections below this
BACKGROUND_LABELS = {'background', 'empty'}   # labels that mean "no item"
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Raspberry Pi Foundation](https://raspberrypi.com)
- [picamera2 library](https://github.com/raspberrypi/picamera2)
- [hx711 Python library](https://github.com/tatobari/hx711py)
- [Flask](https://flask.palletsprojects.com)
- [scikit-learn](https://scikit-learn.org)
- [openpyxl](https://openpyxl.readthedocs.io)

---

<p align="center">Made with ❤ for small vendors everywhere</p>
