"""
app.py
Flask entry point for the Smart Retail Checkout system.

Routes:
  GET  /                  → main checkout UI
  POST /upload            → classify image, return top-5 predictions + bill
  POST /confirm_product   → user confirms / overrides the product choice
  GET  /payment           → payment page
  GET  /payment/success   → success page
  POST /download_bill     → generate & return PDF receipt
"""
from __future__ import annotations

import re
import json
import logging
import os
import queue
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_file

from bill_generator import generate_pdf
from classifier import ZeroShotClassifier, load_labels, save_labels, MODEL_REGISTRY, load_model_config
from product_db import ProductDatabase

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App & config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.resolve()
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

# ---------------------------------------------------------------------------
# Shared services (loaded once at startup)
# ---------------------------------------------------------------------------
logger.info("Initialising classifier …")
classifier = ZeroShotClassifier(
    quantize=os.environ.get("QUANTIZE", "false").lower() == "true",
    ov_device=os.environ.get("OV_DEVICE", "GPU"),
)
UNKNOWN_THRESHOLD = float(os.environ.get("UNKNOWN_THRESHOLD", 40.0))

logger.info("Loading product database …")
product_db = ProductDatabase(BASE_DIR / "product_prices.csv")

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    return send_file(BASE_DIR / "templates" / "webpage_design.html")


@app.route("/upload", methods=["POST"])
def upload_image():
    """
    Accepts a multipart image, runs classification, and returns:
    {
      "top5": [
        {"label": "…", "short_name": "…", "confidence": 91.2},
        …
      ],
      "bill_item": {"Product": "…", "Quantity": 1, "Unit_Price": …, "Total": …} | null
    }
    """
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    try:
        # Save & decode image
        save_path = UPLOAD_FOLDER / file.filename
        file.save(save_path)
        img = cv2.imread(str(save_path))
        if img is None:
            return jsonify({"error": "Could not decode image"}), 422

        # Run zero-shot classification
        predictions = classifier.classify(img, threshold=UNKNOWN_THRESHOLD)

        is_unknown = predictions[0].is_unknown
        top5 = [
            {
                "label": p.label,
                "short_name": p.short_name,
                "confidence": p.confidence,
            }
            for p in predictions
        ]

        # Suppress bill item when confidence is below threshold
        if is_unknown:
            bill_item = None
        else:
            best = predictions[0]
            bill_item = product_db.build_bill(best.short_name)

        return jsonify({
            "top5": top5,
            "bill_item": bill_item,
            "is_unknown": is_unknown,
            "threshold": UNKNOWN_THRESHOLD,
        })

    except Exception as exc:
        logger.exception("Error in /upload")
        return jsonify({"error": str(exc)}), 500


@app.route("/confirm_product", methods=["POST"])
def confirm_product():
    """
    Called when the user overrides the top-1 prediction with one of the
    other top-5 candidates.

    Request JSON: { "short_name": "Granny-Smith" }
    Response:     { "bill_item": { … } }
    """
    data = request.get_json(force=True)
    short_name = (data or {}).get("short_name", "")
    if not short_name:
        return jsonify({"error": "short_name is required"}), 400

    bill_item = product_db.build_bill(short_name)
    if bill_item is None:
        return jsonify({"error": f"Product '{short_name}' not found in database"}), 404

    return jsonify({"bill_item": bill_item})


@app.route("/payment")
def payment_page():
    return send_file(BASE_DIR / "templates" / "payment.html")


@app.route("/payment/success")
def payment_success():
    return send_file(BASE_DIR / "templates" / "success.html")


@app.route("/download_bill", methods=["POST"])
def download_bill():
    """
    Request JSON: { "bill": [...], "total_price": float, "total_items": int }
    Returns: PDF file attachment
    """
    data = request.get_json(force=True) or {}
    bill = data.get("bill", [])
    total_price = float(data.get("total_price", 0))
    total_items = int(data.get("total_items", 0))
    subtotal = data.get("subtotal")
    sst      = data.get("sst")
    if subtotal is not None:
        subtotal = float(subtotal)
    if sst is not None:
        sst = float(sst)

    if not bill:
        return jsonify({"error": "No bill items provided"}), 400

    pdf_buffer = generate_pdf(bill, total_price, total_items, subtotal=subtotal, sst=sst)
    return send_file(
        pdf_buffer,
        as_attachment=True,
        download_name="receipt.pdf",
        mimetype="application/pdf",
    )


# ---------------------------------------------------------------------------
# Label management routes
# ---------------------------------------------------------------------------

# Label format: at least 2 segments separated by '/', each segment is
# alphanumeric (plus hyphens), e.g. "Fruit/Apple/Granny-Smith".
_LABEL_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9-]*(/[A-Za-z0-9][A-Za-z0-9-]*)+$')


def _validate_label(label: str) -> str | None:
    """Return an error message if *label* is invalid, else None."""
    if not label:
        return "label is required"
    if not _LABEL_RE.match(label):
        return (
            "Invalid label format. Use Category/SubCategory/Name with "
            "alphanumeric characters and hyphens (e.g. Fruit/Apple/Fuji)."
        )
    return None

@app.route("/labels", methods=["GET"])
def get_labels():
    """Return the current product label list."""
    return jsonify({"labels": load_labels()})


@app.route("/labels", methods=["POST"])
def add_label():
    """
    Add a new label. Request JSON: { "label": "Category/SubCat/Name" }
    Optionally set "regenerate": true to rebuild weights immediately.
    """
    data = request.get_json(force=True) or {}
    label = (data.get("label") or "").strip()
    err = _validate_label(label)
    if err:
        return jsonify({"error": err}), 400

    labels = load_labels()
    if label in labels:
        return jsonify({"error": "Label already exists"}), 409

    labels.append(label)
    save_labels(labels)

    if data.get("regenerate"):
        classifier.add_label_weight(label)

    return jsonify({"labels": labels, "added": label})


@app.route("/labels", methods=["PUT"])
def modify_label():
    """
    Modify an existing label.
    Request JSON: { "old_label": "…", "new_label": "…" }
    Optionally set "regenerate": true to rebuild weights immediately.
    """
    data = request.get_json(force=True) or {}
    old_label = (data.get("old_label") or "").strip()
    new_label = (data.get("new_label") or "").strip()
    if not old_label:
        return jsonify({"error": "old_label is required"}), 400
    err = _validate_label(new_label)
    if err:
        return jsonify({"error": err}), 400

    labels = load_labels()
    if old_label not in labels:
        return jsonify({"error": f"Label '{old_label}' not found"}), 404
    if new_label in labels:
        return jsonify({"error": f"Label '{new_label}' already exists"}), 409

    idx = labels.index(old_label)
    labels[idx] = new_label
    save_labels(labels)

    if data.get("regenerate"):
        classifier.update_label_weight(old_label, new_label)

    return jsonify({"labels": labels, "modified": {"old": old_label, "new": new_label}})


@app.route("/labels", methods=["DELETE"])
def delete_label():
    """
    Delete a label. Request JSON: { "label": "…" }
    Optionally set "regenerate": true to rebuild weights immediately.
    """
    data = request.get_json(force=True) or {}
    label = (data.get("label") or "").strip()
    if not label:
        return jsonify({"error": "label is required"}), 400

    labels = load_labels()
    if label not in labels:
        return jsonify({"error": f"Label '{label}' not found"}), 404

    labels.remove(label)
    save_labels(labels)

    if data.get("regenerate"):
        classifier.remove_label_weight(label)

    return jsonify({"labels": labels, "deleted": label})


@app.route("/labels/regenerate", methods=["POST"])
def regenerate_weights():
    """Force regeneration of zero-shot classifier weights from current labels."""
    try:
        classifier.rebuild_weights()
        return jsonify({"status": "ok", "message": "Weights regenerated successfully"})
    except Exception as exc:
        logger.exception("Error regenerating weights")
        return jsonify({"error": str(exc)}), 500


@app.route("/labels/regenerate_stream")
def regenerate_weights_stream():
    """SSE endpoint that streams progress while regenerating weights."""
    progress_queue: queue.Queue = queue.Queue()

    def progress_cb(current: int, total: int, label: str) -> None:
        progress_queue.put({"current": current, "total": total, "label": label})

    def generate():
        import threading
        error_holder: list = []

        def do_rebuild():
            try:
                classifier.rebuild_weights(progress_cb=progress_cb)
            except Exception as exc:
                error_holder.append(str(exc))
            finally:
                progress_queue.put(None)  # sentinel

        t = threading.Thread(target=do_rebuild, daemon=True)
        t.start()

        while True:
            item = progress_queue.get()
            if item is None:
                if error_holder:
                    yield f"data: {json.dumps({'error': error_holder[0]})}\n\n"
                else:
                    yield f"data: {json.dumps({'done': True})}\n\n"
                break
            yield f"data: {json.dumps(item)}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------------
# Model selection routes
# ---------------------------------------------------------------------------

@app.route("/model", methods=["GET"])
def get_model():
    """Return available models and the currently active one."""
    active = load_model_config()
    models = [
        {"key": k, "display_name": v["display_name"], "active": k == active}
        for k, v in MODEL_REGISTRY.items()
    ]
    return jsonify({
        "models": models,
        "active": active,
        "ov_device": classifier.ov_device,
        "precision": "INT8" if classifier.quantize else "FP16",
    })


@app.route("/model/precision_stream")
def switch_precision_stream():
    """SSE endpoint: switch OV inference precision between FP16 and INT8."""
    precision = request.args.get("precision", "").strip().upper()
    if precision not in ("FP16", "INT8"):
        return jsonify({"error": "precision must be FP16 or INT8"}), 400
    quantize = precision == "INT8"
    if quantize == classifier.quantize:
        return jsonify({"error": f"Already using {precision}"}), 409

    def generate():
        try:
            classifier.switch_precision(quantize)
            yield f"data: {json.dumps({'done': True, 'precision': precision})}\n\n"
        except Exception as exc:
            logger.exception("Error switching precision")
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/model/switch_stream")
def switch_model_stream():
    """SSE endpoint: switch model and stream regeneration progress."""
    model_key = request.args.get("model", "").strip()
    if not model_key or model_key not in MODEL_REGISTRY:
        return jsonify({"error": f"Unknown model key: {model_key}"}), 400

    if model_key == load_model_config():
        return jsonify({"error": "Already using this model"}), 409

    progress_queue: queue.Queue = queue.Queue()

    def progress_cb(current: int, total: int, label: str) -> None:
        progress_queue.put({"current": current, "total": total, "label": label})

    def generate():
        import threading
        error_holder: list = []

        def do_switch():
            try:
                classifier.switch_model(model_key, progress_cb=progress_cb)
            except Exception as exc:
                error_holder.append(str(exc))
            finally:
                progress_queue.put(None)

        t = threading.Thread(target=do_switch, daemon=True)
        t.start()

        while True:
            item = progress_queue.get()
            if item is None:
                if error_holder:
                    yield f"data: {json.dumps({'error': error_holder[0]})}\n\n"
                else:
                    yield f"data: {json.dumps({'done': True, 'model': model_key})}\n\n"
                break
            yield f"data: {json.dumps(item)}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------------
# Price management routes
# ---------------------------------------------------------------------------

@app.route("/prices", methods=["GET"])
def get_prices():
    """Return all products and their prices."""
    return jsonify({"products": product_db.list_products()})


@app.route("/prices", methods=["POST"])
def add_price():
    """
    Add a new product price.
    Request JSON: { "product": "Granny-Smith", "price": 3.50 }
    """
    data = request.get_json(force=True) or {}
    product = (data.get("product") or "").strip()
    price = data.get("price")

    if not product:
        return jsonify({"error": "product is required"}), 400
    if price is None:
        return jsonify({"error": "price is required"}), 400
    try:
        price = float(price)
    except (TypeError, ValueError):
        return jsonify({"error": "price must be a number"}), 400
    if price < 0:
        return jsonify({"error": "price must not be negative"}), 400

    if not product_db.add_product(product, price):
        return jsonify({"error": f"Product '{product}' already exists"}), 409

    return jsonify({"product": product, "price": round(price, 2)}), 201


@app.route("/prices", methods=["PUT"])
def update_price():
    """
    Update a product's price.
    Request JSON: { "product": "Milk", "price": 2.50 }
    """
    data = request.get_json(force=True) or {}
    product = (data.get("product") or "").strip()
    price = data.get("price")

    if not product:
        return jsonify({"error": "product is required"}), 400
    if price is None:
        return jsonify({"error": "price is required"}), 400
    try:
        price = float(price)
    except (TypeError, ValueError):
        return jsonify({"error": "price must be a number"}), 400
    if price < 0:
        return jsonify({"error": "price must not be negative"}), 400

    if not product_db.update_price(product, price):
        return jsonify({"error": f"Product '{product}' not found in database"}), 404

    return jsonify({"product": product, "price": round(price, 2)})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting Automatic Vision-based Checkout server on port %d …", port)
    app.run(host="0.0.0.0", port=port, debug=False)
