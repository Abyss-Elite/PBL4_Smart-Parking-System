from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import easyocr
import re
import numpy as np
import os
import time
import requests

app = Flask(__name__)
model = YOLO("AI/best.pt")
reader = easyocr.Reader(["en"], gpu=False)

# Tạo các folder nếu chưa có
os.makedirs("AI/crop_debug", exist_ok=True)
os.makedirs("AI/receive", exist_ok=True)


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


def send_plate_to_backend(plate):
    backend_url = "http://192.168.1.22:8083/api/car/LicensePlateNumber"  # đổi thành URL backend thực tế
    data = {"plate": plate}
    try:
        resp = requests.post(backend_url, json=data, timeout=5)
        if resp.status_code == 200:
            print("✅ Gửi plate thành công:", resp.json())
        else:
            print("❌ Lỗi gửi plate, status:", resp.status_code, resp.text)
    except Exception as e:
        print("❌ Exception khi gửi plate:", e)


@app.route("/test_local", methods=["GET"])
def test_local():
    """Test AI với ảnh có sẵn trong thư mục image_test"""
    local_path = "AI/image_test/image2.jpg"  # ← đường dẫn ảnh test
    img = cv2.imread(local_path)
    if img is None:
        return jsonify({"error": "Không tìm thấy ảnh test"}), 400

    timestamp = int(time.time())
    print(f"🧠 Testing local image: {local_path}")

    results = model(img)
    all_ocr_texts = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            debug_path = f"AI/crop_debug/debug_{timestamp}_{x1}_{y1}.jpg"
            cv2.imwrite(debug_path, gray)

            ocr_results = reader.readtext(gray, detail=0, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.")
            if ocr_results:
                box_text = "".join(ocr_results)
                all_ocr_texts.append(box_text)

    combined_text = ".".join(all_ocr_texts) if all_ocr_texts else ""
    plate_text = normalize_plate(combined_text)

    if plate_text:
        print("✅ Biển số:", plate_text)
        send_plate_to_backend(plate_text)
        return jsonify({"plate": plate_text})
    else:
        plate_text = "123456"
        send_plate_to_backend(plate_text)
        return jsonify({"error": plate_text}), 400

@app.route("/upload", methods=["POST"])
def detect_plate():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    # --- Đọc ảnh bytes ---
    img_bytes = file.read()

    # --- Lưu ảnh gốc nhận được ---
    timestamp = int(time.time())
    receive_path = f"AI/receive/receive_{timestamp}.jpg"
    with open(receive_path, "wb") as f:
        f.write(img_bytes)
    print(f"📁 Saved received image to {receive_path}")

    # --- Decode ảnh để xử lý OCR ---
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    results = model(img)

    # Ghép tất cả OCR từ các box YOLO
    all_ocr_texts = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Lưu ảnh crop debug với timestamp
            debug_path = f"AI/crop_debug/debug_{timestamp}_{x1}_{y1}.jpg"
            cv2.imwrite(debug_path, gray)

            ocr_results = reader.readtext(gray, detail=0, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.")
            if ocr_results:
                # Ghép các dòng OCR trong 1 box
                box_text = "".join(ocr_results)
                all_ocr_texts.append(box_text)

    # Ghép tất cả box lại thành 1 chuỗi duy nhất
    combined_text = ".".join(all_ocr_texts) if all_ocr_texts else ""
    plate_text = normalize_plate(combined_text)

    if plate_text:
        print("✅ Biển số:", plate_text)
        send_plate_to_backend(plate_text)
        return jsonify({"plate": plate_text})
    else:
        # print("❌ Không đọc được biển số")
        plate_text= "123456"
        send_plate_to_backend(plate_text)
        print("❌ Không đọc được biển số", plate_text)
        return jsonify({"error": plate_text}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)




# from flask import Flask, request, jsonify
# from ultralytics import YOLO
# import cv2
# import easyocr
# import re
# import numpy as np
# import os
# import time

# app = Flask(__name__)
# model = YOLO("AI/best.pt")
# reader = easyocr.Reader(["en"], gpu=False)

# # Tạo các folder nếu chưa có
# os.makedirs("AI/crop_debug", exist_ok=True)
# os.makedirs("AI/receive", exist_ok=True)

# def normalize_plate(ocr_results):
#     if not ocr_results:
#         return None

#     plate = "".join(ocr_results).upper()
#     plate = plate.replace("O", "0").replace("B", "8").replace("I", "1").replace("T", "1")
#     plate = re.sub(r"[^A-Z0-9.]", "", plate)

#     m = re.match(r"^(\d{2}[A-Z]{1,2})[-.]?(\d{3})\.?(\d{2})$", plate)
#     if m:
#         return f"{m.group(1)}-{m.group(2)}.{m.group(3)}"

#     m2 = re.match(r"^(\d{2}[A-Z]{1,2})[-.]?(\d{4})$", plate)
#     if m2:
#         return f"{m2.group(1)}-{m2.group(2)}"

#     m3 = re.match(r"^(\d{2}[A-Z]{1,2})(\d{3,4}\.?\d{2})$", plate)
#     if m3:
#         part1 = m3.group(1)
#         part2 = m3.group(2)
#         if '.' in part2:
#             return f"{part1}-{part2}"
#         else:
#             return f"{part1}-{part2[:-2]}.{part2[-2:]}"
#     return None

# @app.route("/upload", methods=["POST"])
# def detect_plate():
#     file = request.files.get("file")
#     if not file:
#         return jsonify({"error": "No file uploaded"}), 400

#     # --- Đọc ảnh bytes ---
#     img_bytes = file.read()

#     # --- Lưu ảnh gốc nhận được ---
#     timestamp = int(time.time())
#     receive_path = f"AI/receive/receive_{timestamp}.jpg"
#     with open(receive_path, "wb") as f:
#         f.write(img_bytes)
#     print(f"📁 Saved received image to {receive_path}")
#     # --------------------------------------

#     # --- Decode ảnh để xử lý OCR ---
#     img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
#     results = model(img)

#     plate_text = None
#     for r in results:
#         for box in r.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             crop = img[y1:y2, x1:x2]

#             # Chuyển sang grayscale và xử lý
#             gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
#             gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
#             _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#             # Lưu ảnh crop debug
#             cv2.imwrite("AI/crop_debug/temp.jpg", gray)

#             # OCR
#             ocr_results = reader.readtext(gray, detail=0, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.")
#             plate = normalize_plate(ocr_results)
#             if plate:
#                 plate_text = plate
#                 break
#         if plate_text:
#             break

#     if plate_text:
#         print("✅ Biển số:", plate_text)
#         return jsonify({"plate": plate_text})
#     else:
#         print("❌ Không đọc được biển số")
#         return jsonify({"error": "Không đọc được biển số"}), 400

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)




















# from ultralytics import YOLO
# import cv2
# import easyocr
# import re

# model = YOLO("AI/best.pt")

# img_path = "AI/image_test/receive_1760090162.jpg"
# results = model(img_path)

# reader = easyocr.Reader(["en"], gpu=False)

# def normalize_plate(ocr_results):
#     if not ocr_results:
#         return None

#     plate = "".join(ocr_results).upper()

#     plate = plate.replace("O", "0").replace("B", "8").replace("I", "1")

#     plate = re.sub(r"[^A-Z0-9.]", "", plate)

#     m = re.match(r"^(\d{2}[A-Z]{1,2})[-.]?(\d{3})\.?(\d{2})$", plate)
#     if m:
#         return f"{m.group(1)}-{m.group(2)}.{m.group(3)}"

#     m2 = re.match(r"^(\d{2}[A-Z]{1,2})[-.]?(\d{4})$", plate)
#     if m2:
#         return f"{m2.group(1)}-{m2.group(2)}"

#     m3 = re.match(r"^(\d{2}[A-Z]{1,2})(\d{3,4}\.?\d{2})$", plate)
#     if m3:
#         part1 = m3.group(1)
#         part2 = m3.group(2)
#         if '.' in part2:
#             return f"{part1}-{part2}"
#         else:
#             return f"{part1}-{part2[:-2]}.{part2[-2:]}"
#     return None

# for r in results:
#     for box in r.boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         img = cv2.imread(img_path)
#         crop = img[y1:y2, x1:x2]

#         gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
#         gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
#         _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#         cv2.imwrite("AI/crop_debug.jpg", gray)

#         ocr_results = reader.readtext(gray, detail=0, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.")
#         print(ocr_results)
#         plate = normalize_plate(ocr_results)
#         if plate:
#             print("Biển số chuẩn hóa:", plate)
#         else:
#             print("OCR lỗi:", ocr_results)