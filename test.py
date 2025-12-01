import onnxruntime as ort
import numpy as np
import cv2
import time
import os

# ======= Cấu hình =======

ONNX_MODEL_PATH = "traffic_sign_model.onnx"
IMG_FOLDER = "/home/minhthong/Desktop/code/self-driving/test_data"
INPUT_SIZE = 640
CONF_THRESH = 0.25
IOU_THRESH = 0.45
classes = ['go-ahead', 'stop', 'turn-around', 'turn-left', 'turn-right']

# ======= Load ONNX model =======

session = ort.InferenceSession(
    ONNX_MODEL_PATH,
    providers=['CUDAExecutionProvider']   # nếu có GPU thì thay bằng ['CUDAExecutionProvider']
)

input_name = session.get_inputs()[0].name


# ======= Hàm tiền xử lý =======

def preprocess(img):
    h, w = img.shape[:2]
    scale = INPUT_SIZE / max(h, w)
    nh, nw = int(h * scale), int(w * scale)

    img_resized = cv2.resize(img, (nw, nh))

    pad_x = (INPUT_SIZE - nw) // 2
    pad_y = (INPUT_SIZE - nh) // 2

    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = img_resized

    img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_rgb = np.transpose(img_rgb, (2, 0, 1))
    img_rgb = np.expand_dims(img_rgb, axis=0)
    return img_rgb


# ======= Lấy danh sách ảnh =======

image_files = [
    f for f in os.listdir(IMG_FOLDER)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
]

total_time = 0
count = 0

# ======= Vòng lặp xử lý =======

for img_name in image_files:
    img_path = os.path.join(IMG_FOLDER, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Không thể load: {img_name}")
        continue

    # ---- Preprocess ----
    img_input = preprocess(img)

    # ---- Inference + đo thời gian ----
    start_time = time.time()
    outputs = session.run(None, {input_name: img_input})
    end_time = time.time()

    infer_time = (end_time - start_time) * 1000   # đổi ms
    total_time += infer_time
    count += 1

    print(f"{img_name}: {infer_time:.2f} ms")


# ======= Tổng kết =======

print("\n============================")
print(f"Số ảnh xử lý: {count}")
print(f"Tổng thời gian: {total_time:.2f} ms")

if count > 0:
    avg = total_time / count
    print(f"Thời gian trung bình mỗi ảnh: {avg:.2f} ms")
    print(f"FPS: {1000 / avg:.2f}")
print("============================")
