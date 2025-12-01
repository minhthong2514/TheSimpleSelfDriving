import onnxruntime as ort
import numpy as np
import cv2
import time

# ========================
#      CONFIG
# ========================

ONNX_MODEL_PATH = "traffic_sign_model.onnx"
INPUT_SIZE = 640
CONF_THRESH = 0.7
IOU_THRESH = 0.45

classes = ['go-ahead', 'stop', 'turn-around', 'turn-left', 'turn-right']


# ========================
#  LOAD ONNX (GPU/CPU)
# ========================

providers = ort.get_available_providers()
print("Available providers:", providers)

if "CUDAExecutionProvider" in providers:
    session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CUDAExecutionProvider"])
    print(">>> Using GPU for inference.")
else:
    session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
    print(">>> GPU NOT available → using CPU.")

input_name = session.get_inputs()[0].name


# ========================
#  PREPROCESS
# ========================

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

    return img_rgb, scale, pad_x, pad_y, nw, nh


# ========================
#  NMS (OpenCV)
# ========================

def nms(boxes, scores, iou_thresh):
    if len(boxes) == 0:
        return []

    idxs = cv2.dnn.NMSBoxes(
        bboxes=boxes,
        scores=scores,
        score_threshold=CONF_THRESH,
        nms_threshold=iou_thresh
    )

    return idxs.flatten() if len(idxs) > 0 else []


# ========================
#  CAMERA
# ========================

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Không thể mở camera!")
    exit()

print("\n===== Camera started =====")


# ========================
#  MAIN LOOP
# ========================

fps_smooth = 0   # FPS mượt hơn (EMA)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không lấy được frame!")
        break

    img_input, scale, pad_x, pad_y, nw, nh = preprocess(frame)

    # -----------------------------
    # Inference + FPS chuẩn
    # -----------------------------
    start = time.time()
    outputs = session.run(None, {input_name: img_input})
    infer_time = (time.time() - start) * 1000  # ms

    fps = 1000.0 / infer_time
    fps_smooth = fps_smooth * 0.9 + fps * 0.1   # smoothing

    pred = outputs[0][0]  # (num_boxes, 85)

    # ---- Decode YOLO ----
    boxes = []
    scores = []
    class_ids = []

    for det in pred:
        conf = det[4]
        if conf < CONF_THRESH:
            continue

        class_prob = det[5:]
        class_id = int(np.argmax(class_prob))
        score = float(conf * class_prob[class_id])
        if score < CONF_THRESH:
            continue

        cx, cy, w, h = det[:4]

        # map ngược về ảnh gốc
        x1 = int((cx - w/2 - pad_x) / scale)
        y1 = int((cy - h/2 - pad_y) / scale)
        x2 = int((cx + w/2 - pad_x) / scale)
        y2 = int((cy + h/2 - pad_y) / scale)

        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(score)
        class_ids.append(class_id)

    # ---- NMS ----
    idxs = nms(boxes, scores, IOU_THRESH)

    # ---- Draw ----
    for i in idxs:
        x, y, w, h = boxes[i]
        label = classes[class_ids[i]]
        score = scores[i]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {score:.2f}",
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)

    # ---- Text (FPS chuẩn) ----
    cv2.putText(frame, f"Infer: {infer_time:.1f} ms | FPS: {fps_smooth:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0), 2)

    cv2.imshow("YOLO ONNX Camera", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
