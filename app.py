import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from flask import send_from_directory

import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

try:
    import cv2
except Exception:
    cv2 = None

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_mobilenetv2_real_fake.keras")
CFG_PATH = os.path.join(MODEL_DIR, "eval_config.json")

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

IMG_SIZE = 160

ALLOWED_IMAGE_EXT = {"jpg", "jpeg", "png", "webp"}
ALLOWED_VIDEO_EXT = {"mp4", "mov", "avi", "mkv"}

MAX_CONTENT_LENGTH = 200 * 1024 * 1024

DEFAULT_MAX_FRAMES = 12
DEFAULT_STRIDE = 0  


app = Flask(__name__)
app.config["SECRET_KEY"] = "dev-secret-key-change-me"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///hoaxbuster.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

CORS(app)
db = SQLAlchemy(app)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    media_type = db.Column(db.String(20), nullable=False)  
    label = db.Column(db.String(20), nullable=False)       
    fake_prob = db.Column(db.Float, nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    threshold = db.Column(db.Float, nullable=False)
    model_version = db.Column(db.String(100), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    extra_json = db.Column(db.Text, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        extra = {}
        if self.extra_json:
            try:
                extra = json.loads(self.extra_json)
            except Exception:
                extra = {"raw": self.extra_json}

        return {
            "id": self.id,
            "filename": self.filename,
            "media_type": self.media_type,
            "label": self.label,
            "fake_prob": float(self.fake_prob),
            "confidence": float(self.confidence),
            "threshold": float(self.threshold),
            "model_version": self.model_version,
            "created_at": self.created_at.isoformat() + "Z",
            "extra": extra,
        }


with app.app_context():
    db.create_all()

def load_eval_config() -> Dict[str, Any]:
    if not os.path.exists(CFG_PATH):
        return {"threshold": 0.5, "label_real": "REAL", "label_fake": "FAKE", "model_version": "v1"}
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


CFG = load_eval_config()
THRESHOLD = float(CFG.get("threshold", 0.5))
LABEL_REAL = str(CFG.get("label_real", "REAL"))
LABEL_FAKE = str(CFG.get("label_fake", "FAKE"))
MODEL_VERSION = str(CFG.get("model_version", "mobilenetv2_real_fake"))


_model: Optional[tf.keras.Model] = None

def get_model() -> tf.keras.Model:
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")
        _model = load_model(MODEL_PATH)
    return _model

def allowed_file(filename: str) -> bool:
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return (ext in ALLOWED_IMAGE_EXT) or (ext in ALLOWED_VIDEO_EXT)


def get_media_type(filename: str) -> str:
    ext = filename.rsplit(".", 1)[1].lower()
    if ext in ALLOWED_IMAGE_EXT:
        return "image"
    if ext in ALLOWED_VIDEO_EXT:
        return "video"
    return "unknown"


def preprocess_rgb_frame(frame_rgb: np.ndarray) -> np.ndarray:
    """
    frame_rgb: (H, W, 3) uint8, RGB
    output: (1, 160, 160, 3) float32, preprocess_input MobileNetV2
    """
    frame_resized = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    x = frame_resized.astype("float32")
    x = preprocess_input(x)  
    x = np.expand_dims(x, axis=0)
    return x


def decide_from_fake_prob(fake_prob: float) -> Tuple[str, float]:
    is_fake = fake_prob >= THRESHOLD
    if is_fake:
        return LABEL_FAKE, float(fake_prob)
    return LABEL_REAL, float(1.0 - fake_prob)


def predict_image(file_path: str) -> Dict[str, Any]:
    if cv2 is None:
        raise RuntimeError("opencv-python belum terpasang. Install: pip install opencv-python")

    bgr = cv2.imread(file_path)
    if bgr is None:
        raise ValueError("Gagal membaca gambar. Pastikan file gambar valid.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    x = preprocess_rgb_frame(rgb)
    model = get_model()

    t0 = time.time()
    fake_prob = float(model.predict(x, verbose=0)[0][0])
    latency_ms = int((time.time() - t0) * 1000)

    label, confidence = decide_from_fake_prob(fake_prob)

    return {
        "media_type": "image",
        "fake_prob": fake_prob,
        "label": label,
        "confidence": confidence,
        "threshold": THRESHOLD,
        "latency_ms": latency_ms,
        "model_version": MODEL_VERSION,
    }


def sample_video_frames(video_path: str, max_frames: int = DEFAULT_MAX_FRAMES) -> List[Tuple[int, np.ndarray]]:
    if cv2 is None:
        raise RuntimeError("opencv-python belum terpasang. Install: pip install opencv-python")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Gagal membuka video. Pastikan file video valid.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    if total_frames <= 0:
        frames = []
        idx = 0
        while len(frames) < max_frames:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append((idx, frame_rgb))
            idx += 1
        cap.release()
        return frames

    if total_frames <= max_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, num=max_frames, dtype=int).tolist()

    sampled = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        sampled.append((int(i), frame_rgb))

    cap.release()
    return sampled


def predict_video(video_path: str, max_frames: int = DEFAULT_MAX_FRAMES) -> Dict[str, Any]:
    frames = sample_video_frames(video_path, max_frames=max_frames)
    if len(frames) == 0:
        raise ValueError("Tidak ada frame yang berhasil diambil dari video.")

    model = get_model()

    t0 = time.time()
    frame_scores = []
    for idx, rgb in frames:
        x = preprocess_rgb_frame(rgb)
        fake_prob = float(model.predict(x, verbose=0)[0][0])
        fake_prob = 1.0 - real_prob
        frame_scores.append({"frame_index": idx, "fake_prob": fake_prob})

    latency_ms = int((time.time() - t0) * 1000)

    probs = [s["fake_prob"] for s in frame_scores]
    agg_fake_prob = float(np.mean(probs))

    label, confidence = decide_from_fake_prob(agg_fake_prob)

    topk = sorted(frame_scores, key=lambda x: x["fake_prob"], reverse=True)[:3]

    return {
        "media_type": "video",
        "fake_prob": agg_fake_prob,
        "label": label,
        "confidence": confidence,
        "threshold": THRESHOLD,
        "latency_ms": latency_ms,
        "model_version": MODEL_VERSION,
        "frame_scores": frame_scores,
        "top_suspicious_frames": topk,
        "sampled_frames_count": len(frames),
    }

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "time": datetime.utcnow().isoformat() + "Z",
        "model_loaded": _model is not None,
        "model_path": os.path.relpath(MODEL_PATH, BASE_DIR),
    })


@app.route("/api/config", methods=["GET"])
def get_config():
    return jsonify({
        "img_size": IMG_SIZE,
        "threshold": THRESHOLD,
        "label_real": LABEL_REAL,
        "label_fake": LABEL_FAKE,
        "model_version": MODEL_VERSION,
        "allowed_image_ext": sorted(list(ALLOWED_IMAGE_EXT)),
        "allowed_video_ext": sorted(list(ALLOWED_VIDEO_EXT)),
        "max_upload_mb": int(MAX_CONTENT_LENGTH / (1024 * 1024)),
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Field 'file' tidak ditemukan."}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Nama file kosong."}), 400

    if not allowed_file(f.filename):
        return jsonify({"error": "Format file tidak didukung."}), 400

    filename = secure_filename(f.filename)
    ts = int(time.time())
    save_name = f"{ts}_{filename}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], save_name)
    f.save(save_path)

    media_type = get_media_type(save_name)

    try:
        if media_type == "image":
            result = predict_image(save_path)
            extra = {}
        elif media_type == "video":
            max_frames = request.form.get("max_frames", type=int) or DEFAULT_MAX_FRAMES
            max_frames = max(1, min(max_frames, 60))  # batas aman
            result = predict_video(save_path, max_frames=max_frames)
            extra = {
                "frame_scores": result.get("frame_scores", []),
                "top_suspicious_frames": result.get("top_suspicious_frames", []),
                "sampled_frames_count": result.get("sampled_frames_count", 0),
            }
        else:
            return jsonify({"error": "Tipe media tidak dikenali."}), 400

        pred = Prediction(
            filename=save_name,
            media_type=media_type,
            label=result["label"],
            fake_prob=float(result["fake_prob"]),
            confidence=float(result["confidence"]),
            threshold=float(result["threshold"]),
            model_version=result.get("model_version"),
            extra_json=json.dumps(extra) if extra else None
        )
        db.session.add(pred)
        db.session.commit()

        payload = {
            "id": pred.id,
            "media_type": media_type,
            "filename": save_name,
            "result": {
                "label": result["label"],
                "confidence": float(result["confidence"]),
                "fake_prob": float(result["fake_prob"]),
                "threshold": float(result["threshold"]),
            },
            "model": {
                "version": result.get("model_version"),
                "img_size": IMG_SIZE,
            },
            "timing": {
                "latency_ms": int(result.get("latency_ms", 0)),
            },
            "created_at": pred.created_at.isoformat() + "Z",
            "artifacts": extra if extra else {
                "frame_scores": [],
                "top_suspicious_frames": []
            }
        }

        return jsonify(payload), 200


    except Exception as e:
        try:
            if os.path.exists(save_path):
                os.remove(save_path)
        except Exception:
            pass

        return jsonify({"error": str(e)}), 500


@app.route("/api/history", methods=["GET"])
def history():

    limit = request.args.get("limit", type=int) or 20
    limit = max(1, min(limit, 200))

    items = Prediction.query.order_by(Prediction.id.desc()).limit(limit).all()
    return jsonify({
        "count": len(items),
        "items": [it.to_dict() for it in items]
    })


@app.route("/api/history/<int:pred_id>", methods=["GET"])
def history_detail(pred_id: int):
    item = Prediction.query.get_or_404(pred_id)
    return jsonify(item.to_dict())

@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/")
def home():
    return {"status": "ok", "message": "Backend running. Open frontend at http://localhost:5173"}

if __name__ == "__main__":
    try:
        get_model()
        print(f"Model loaded: {MODEL_PATH}")
    except Exception as e:
        print(f"Warning: model belum bisa diload saat startup: {e}")

    app.run(host="127.0.0.1", port=5000, debug=True)