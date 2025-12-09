import onnxruntime as ort
import numpy as np
import cv2
import time
import threading

# ========================
#      CONFIG
# ========================

ONNX_MODEL_PATH = "/home/minhthong/Desktop/code/self-driving/traffic_sign_model.onnx"
INPUT_SIZE = 640
CONF_THRESH = 0.7
IOU_THRESH = 0.45

classes = ['go-ahead', 'stop', 'turn-around', 'turn-left', 'turn-right']

DISPLAY_SKIP = 2   # ✅ chỉ hiển thị 1/2 frame để tránh nghẽn
fps_smooth = 0

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

    return img_rgb, scale, pad_x, pad_y

# ========================
#  NMS
# ========================

def nms(boxes, scores):
    if len(boxes) == 0:
        return []

    idxs = cv2.dnn.NMSBoxes(
        bboxes=boxes,
        scores=scores,
        score_threshold=CONF_THRESH,
        nms_threshold=IOU_THRESH
    )

    return idxs.flatten() if len(idxs) > 0 else []

# ========================
#  CAMERA THREAD
# ========================

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame = None
running = True

def camera_thread():
    global frame, running
    while running:
        ret, img = cap.read()
        if ret:
            frame = img

# ========================
#  INFERENCE THREAD
# ========================

det_frame = None

def infer_thread():
    global frame, det_frame, fps_smooth

    while running:
        if frame is None:
            continue

        img_input, scale, pad_x, pad_y = preprocess(frame)

        # ---- INFERENCE ----
        start = time.time()
        outputs = session.run(None, {input_name: img_input})
        infer_time = (time.time() - start) * 1000  # ms

        fps = 1000.0 / infer_time
        fps_smooth = fps_smooth * 0.9 + fps * 0.1

        pred = outputs[0][0]

        boxes, scores, class_ids = [], [], []

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

            x1 = int((cx - w/2 - pad_x) / scale)
            y1 = int((cy - h/2 - pad_y) / scale)
            x2 = int((cx + w/2 - pad_x) / scale)
            y2 = int((cy + h/2 - pad_y) / scale)

            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(score)
            class_ids.append(class_id)

        idxs = nms(boxes, scores)

        draw = frame.copy()

        for i in idxs:
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            score = scores[i]

            cv2.rectangle(draw, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(draw, f"{label} {score:.2f}",
                        (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,255,0), 2)

        cv2.putText(draw, f"Infer: {infer_time:.1f} ms | FPS: {fps_smooth:.1f}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        det_frame = draw

# ========================
#  START THREADS
# ========================

threading.Thread(target=camera_thread, daemon=True).start()
threading.Thread(target=infer_thread, daemon=True).start()

print("\n===== START REALTIME DETECTION =====")

display_count = 0

while True:
    if det_frame is not None:
        display_count += 1

        if display_count % DISPLAY_SKIP == 0:
            cv2.imshow("YOLO ONNX Threaded", det_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        running = False
        break

cap.release()
cv2.destroyAllWindows()
