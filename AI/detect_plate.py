from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
from paddleocr import PaddleOCR
import re
import numpy as np
import os
import time
import requests
import threading
from difflib import SequenceMatcher


app = Flask(__name__)
model = YOLO("best.pt")
reader = PaddleOCR(lang='ch', use_textline_orientation=True)
CAM_URL = "http://192.168.1.14/video"   
PROCESS_INTERVAL = 0.2                  
RECENT_BUFFER_SIZE = 10                 
SIMILARITY_THRESHOLD = 0.85             
SEND_THRESHOLD = 3  

os.makedirs("crop_debug", exist_ok=True)
os.makedirs("receive", exist_ok=True)
_recent_plates = []    
_last_process_time = 0 
is_streaming = False

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def normalize_plate(ocr_text):
    """Chuẩn hóa chuỗi biển số sau khi ghép OCR"""
    if not ocr_text:
        return None

    plate = ocr_text.upper()
    plate = plate.replace("O", "0").replace("B", "8").replace("I", "1").replace("T", "1")
    plate = re.sub(r"[^A-Z0-9.]", "", plate)

    m = re.match(r"^(\d{2}[A-Z]{1,2})[-.]?(\d{3})\.?(\d{2})$", plate)
    if m:
        return f"{m.group(1)}-{m.group(2)}.{m.group(3)}"

    m2 = re.match(r"^(\d{2}[A-Z]{1,2})[-.]?(\d{4})$", plate)
    if m2:
        return f"{m2.group(1)}-{m2.group(2)}"

    m3 = re.match(r"^(\d{2}[A-Z]{1,2})(\d{3,4}\.?\d{2})$", plate)
    if m3:
        part1 = m3.group(1)
        part2 = m3.group(2)
        if '.' in part2:
            return f"{part1}-{part2}"
        else:
            return f"{part1}-{part2[:-2]}.{part2[-2:]}"
    return None

def enhance_image_for_ocr(crop):
    """Tăng chất lượng ảnh để OCR chính xác hơn"""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.8, beta=15)  
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,  
        21, 10
    )
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    return thresh

def send_plate_to_backend(plate):
    backend_url = "http://192.168.1.26:8083/api/car/LicensePlateNumber" 
    data = {"plate": plate}
    try:
        resp = requests.post(backend_url, json=data, timeout=20)
        if resp.status_code == 200:
            print("✅ Gửi plate thành công:", resp.json())
        else:
            print("❌ Lỗi gửi plate, status:", resp.status_code, resp.text)
    except Exception as e:
        print("❌ Exception khi gửi plate:", e)

def update_plate_buffer_and_maybe_send(plate):
    global _recent_plates
    if plate is None:
        return

    _recent_plates.append(plate)
    if len(_recent_plates) > RECENT_BUFFER_SIZE:
        _recent_plates.pop(0)

    groups = []
    for p in _recent_plates:
        placed = False
        for g in groups:
            if similar(p, g[0]) >= SIMILARITY_THRESHOLD:
                g[1].append(p)
                placed = True
                break
        if not placed:
            groups.append([p, [p]])

    if not groups:
        return

    best = max(groups, key=lambda x: len(x[1]))
    rep_string = best[0]
    count = len(best[1])

    if count >= SEND_THRESHOLD:
        print(f"→ Xác nhận biển: {rep_string} (count={count}) — gửi sang backend")
        send_plate_to_backend(rep_string)
        _recent_plates = []

def stream_process():
    global _last_process_time
    cap = cv2.VideoCapture(CAM_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    frame_count = 0
    frames_to_capture = 10  
    plates_detected = []
    if not cap.isOpened():
        print("❌ Không thể mở stream từ ESP32-CAM:", CAM_URL)
        return

    print("🎥 Bắt đầu đọc stream:", CAM_URL)
    while frame_count < frames_to_capture:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Không đọc được frame, thử lại...")
            time.sleep(0.1)
            continue

        frame_count += 1
        if frame is None:
            print("⚠️ Không đọc được frame từ stream (có thể stream bị ngắt)")
            break

        plate_text = process_image(frame, int(time.time()))
        if plate_text:
            plates_detected.append(plate_text)
            print(f"🧾 Frame {frame_count}: {plate_text}")
        time.sleep(0.1)  # 100ms mỗi frame

    cap.release()
    if plates_detected:
        final_plate = max(set(plates_detected), key=plates_detected.count)
        print(f"✅ Biển số xác định: {final_plate}")
        send_plate_to_backend(final_plate)
    else:
        print("❌ Không đọc được biển số nào")
    
    global is_streaming 
    is_streaming = False


def process_image(img, timestamp):
    """Nhận diện biển số, crop, tăng chất lượng và OCR"""
    results = model.predict(img, device='cpu')
    all_ocr_texts = []

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        for (x1, y1, x2, y2) in boxes:

            crop_img = img[y1:y2, x1:x2]
            if crop_img.size == 0:
                continue 

            crop_img = cv2.copyMakeBorder(
                crop_img, 5, 5, 5, 5,
                cv2.BORDER_CONSTANT,
                value=(255, 255, 255)
            )
            
            h, w = crop_img.shape[:2]
            if h < 40 or w < 120:
                scale = 3
            elif h < 80:
                scale = 2
            else:
                scale = 1
            crop_img = cv2.resize(crop_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            enhanced_bgr = enhance_image_for_ocr(crop_img)

            if len(enhanced_bgr.shape) == 2:
                enhanced_bgr = cv2.cvtColor(enhanced_bgr, cv2.COLOR_GRAY2BGR)
            
            debug_path = f"crop_debug/debug_{timestamp}_{x1}_{y1}.jpg"
            cv2.imwrite(debug_path, enhanced_bgr)

            try:
                ocr_results = reader.ocr(enhanced_bgr)
                if ocr_results and len(ocr_results) > 0 and ocr_results[0]:
                    for line in ocr_results[0]:
                        text = line[1][0]
                        all_ocr_texts.append(text)
            except Exception as e:
                print("⚠️ Lỗi khi OCR:", e)
    if all_ocr_texts:
        print("🧾 OCR đọc được:", all_ocr_texts)
        combined_text = ".".join(all_ocr_texts)
    else:
        print("⚠️ Không đọc được text nào từ OCR")
        combined_text = ""
    return normalize_plate(combined_text)


@app.route("/test_local", methods=["GET"])
def test_local():
    local_path = "image_test/image4.jpg"
    img = cv2.imread(local_path)
    if img is None:
        return jsonify({"error": "Không tìm thấy ảnh test"}), 400

    timestamp = int(time.time())
    print(f"🧠 Testing local image: {local_path}")

    plate_text = process_image(img, timestamp)

    if plate_text:
        print("✅ Biển số:", plate_text)
        return jsonify({"plate": plate_text})
    else:
        plate_text = "123456"
        print("❌ Không đọc được biển số", plate_text)
        return jsonify({"error": plate_text}), 400


@app.route("/upload", methods=["POST"])
def detect_plate():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    img_bytes = file.read()
    timestamp = int(time.time())
    receive_path = f"receive/receive_{timestamp}.jpg"
    with open(receive_path, "wb") as f:
        f.write(img_bytes)
    print(f"📁 Saved received image to {receive_path}")

    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    plate_text = process_image(img, timestamp)

    if plate_text:
        print("✅ Biển số:", plate_text)
        send_plate_to_backend(plate_text)
        return jsonify({"plate": plate_text})
    else:
        plate_text = "123456"
        send_plate_to_backend(plate_text)
        print("❌ Không đọc được biển số", plate_text)
        return jsonify({"error": plate_text}), 400

@app.route("/postAI", methods=["GET"])
def postAI():
    global is_streaming
    status = request.args.get("status", "")
    print(f"📩 Nhận tín hiệu từ ESP32: status={status}, is_streaming={is_streaming}")
    
    if status == "CoXe":
        if not is_streaming:
            t = threading.Thread(target=stream_process, daemon=True)
            t.start()
            is_streaming = True
            return jsonify({"message": "Bắt đầu xử lý video"})
        else:
            return jsonify({"message": "Đang xử lý video, bỏ qua yêu cầu mới"})
    else:
        return jsonify({"message": f"Trạng thái không hợp lệ: {status}"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
