from ultralytics import YOLO
import cv2
import easyocr
import re

model = YOLO("AI/best.pt")

img_path = "AI/test.jpg"
results = model(img_path)

reader = easyocr.Reader(["en"], gpu=False)

def normalize_plate(ocr_results):
    if not ocr_results:
        return None

    plate = "".join(ocr_results).upper()

    plate = plate.replace("O", "0").replace("B", "8").replace("I", "1")

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

for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        img = cv2.imread(img_path)
        crop = img[y1:y2, x1:x2]

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        cv2.imwrite("AI/crop_debug.jpg", gray)

        ocr_results = reader.readtext(gray, detail=0, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.")
        print(ocr_results)
        plate = normalize_plate(ocr_results)
        if plate:
            print("Biển số chuẩn hóa:", plate)
        else:
            print("OCR lỗi:", ocr_results)
