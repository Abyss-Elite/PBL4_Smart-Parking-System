import os
import random
import shutil

# Đường dẫn tới dataset
base_path = "./dataset/detection"

images_train = os.path.join(base_path, "images/train")
labels_train = os.path.join(base_path, "labels/train")

images_test = os.path.join(base_path, "images/test")
labels_test = os.path.join(base_path, "labels/test")

# Tạo folder test nếu chưa có
os.makedirs(images_test, exist_ok=True)
os.makedirs(labels_test, exist_ok=True)

# Lấy danh sách file ảnh trong train
image_files = [f for f in os.listdir(images_train) if f.endswith(".jpg") or f.endswith(".png")]

# Số lượng ảnh muốn chuyển sang test (ví dụ 10% train)
num_test = int(len(image_files) * 0.1)

# Random chọn file
test_files = random.sample(image_files, num_test)

for img_file in test_files:
    # file ảnh
    src_img = os.path.join(images_train, img_file)
    dst_img = os.path.join(images_test, img_file)

    # file label (đổi đuôi .jpg/.png thành .txt)
    label_file = os.path.splitext(img_file)[0] + ".txt"
    src_label = os.path.join(labels_train, label_file)
    dst_label = os.path.join(labels_test, label_file)

    # Move ảnh và label
    shutil.move(src_img, dst_img)
    if os.path.exists(src_label):
        shutil.move(src_label, dst_label)

print(f"Đã chuyển {num_test} ảnh và label tương ứng sang thư mục test.")
