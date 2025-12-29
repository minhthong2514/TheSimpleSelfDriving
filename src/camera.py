import cv2
import onnxruntime as ort
import numpy as np
import time

# =========================
#        CONFIG
# =========================

ONNX_MODEL_PATH = "/home/jetson-nano/Desktop/code/Do_an_robot/src/traffic_sign_model.onnx"
INPUT_SIZE = 640

CONF_THRESH = 0.9
IOU_THRESH = 0.45

classes = ['go-ahead', 'stop', 'turn-around', 'turn-left', 'turn-right']
line_detect_mode = 0
error = 0
sign_id = -1
# =========================
#    LOAD ONNX (GPU/CPU)
# =========================

providers = ort.get_available_providers()
print("Available providers:", providers)

if "CUDAExecutionProvider" in providers:
    session = ort.InferenceSession(
        ONNX_MODEL_PATH,
        providers=["CUDAExecutionProvider"]
    )
    print(">>> Using GPU for YOLO")
else:
    session = ort.InferenceSession(
        ONNX_MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )
    print(">>> Using CPU for YOLO")

input_name = session.get_inputs()[0].name

# =========================
#       PREPROCESS
# =========================

def preprocess_yolo(img):
    h, w = img.shape[:2]
    scale = INPUT_SIZE / max(h, w)
    nh, nw = int(h * scale), int(w * scale)

    img_resized = cv2.resize(img, (nw, nh))

    pad_x = (INPUT_SIZE - nw) // 2
    pad_y = (INPUT_SIZE - nh) // 2

    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y+nh, pad_x:pad_x+nw] = img_resized

    img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_rgb = np.transpose(img_rgb, (2,0,1))
    img_rgb = np.expand_dims(img_rgb, axis=0)

    return img_rgb, scale, pad_x, pad_y

# =========================
#         NMS
# =========================

def nms(boxes, scores, iou_thresh=IOU_THRESH):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_thresh]

    return keep

# =========================
#      LINE PROCESSING
# =========================

def detect_line(frame):
    h, w = frame.shape[:2]

    # ========================
    # ROI: nửa dưới ảnh
    # ========================
    roi = frame[int(h * 0.2):h, :]

    # ========================
    # Convert to HSV
    # ========================
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # ========================
    # Mask màu đỏ (2 dải)
    # ========================
    lower_red1 = np.array([0, 80, 80])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 80, 80])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    binary = cv2.bitwise_or(mask1, mask2)

    # ========================
    # Morphology nhẹ
    # ========================
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # ========================
    # Find contours
    # ========================
    cnts = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]

    error = None

    if contours:
        largest = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest) > 500:
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                cv2.circle(roi, (cx, cy), 5, (0,255,0), -1)
                cv2.drawContours(roi, [largest], -1, (0,255,0), 2)

                error = cx - (w // 2)

    return error, roi, binary

# =========================
#          CAMERA
# =========================

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Không mở được camera")
    exit()

print("===== START =====")

# =========================
#        MAIN LOOP
# =========================

last_yolo_time = 0
YOLO_INTERVAL = 0.05   # YOLO chạy ~6–7 FPS

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # ==================================================
    # 1️⃣ YOLO – PHÁT HIỆN BIỂN BÁO (ĐỘC LẬP)
    # ==================================================
    if current_time - last_yolo_time > YOLO_INTERVAL:

        sign_id = -1   # reset MỖI LẦN YOLO CHẠY

        img_input, scale, pad_x, pad_y = preprocess_yolo(frame)
        outputs = session.run(None, {input_name: img_input})
        preds = outputs[0][0]

        boxes, scores, class_ids = [], [], []

        for det in preds:
            conf = det[4]
            if conf < CONF_THRESH:
                continue

            class_probs = det[5:]
            class_id = int(class_probs.argmax())
            score = conf * class_probs[class_id]

            if score < CONF_THRESH:
                continue

            cx, cy, w, h = det[:4]

            x1 = int((cx - w/2 - pad_x) / scale)
            y1 = int((cy - h/2 - pad_y) / scale)
            x2 = int((cx + w/2 - pad_x) / scale)
            y2 = int((cy + h/2 - pad_y) / scale)

            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(score)
            class_ids.append(class_id)

        idxs = nms(boxes, scores)

        if len(idxs) > 0:
            i = idxs[0]
            sign_id = class_ids[i]   # 👈 CHỈ GÁN ID, KHÔNG LOGIC
            detected_sign = classes[sign_id]

            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, str(detected_sign),
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 0, 0), 2)
            # if sign is stop or turn-around or turn-left or turn-down, detect mode is off
            if detected_sign in ['stop', 'turn-around', 'turn-left', 'turn-right']:
                line_detect_mode = 0

            elif detected_sign == 'go-ahead':
                line_detect_mode = 1

        last_yolo_time = current_time

    # ==================================================
    # 2️⃣ LINE FOLLOWING – ĐỘC LẬP HOÀN TOÀN
    # ==================================================
    error, roi, binary = detect_line(frame)
    print(f"State: {line_detect_mode} | Error: {error} | Sign: {sign_id}")
    cv2.imshow("Frame", frame)
    cv2.imshow("Line Binary", binary)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
