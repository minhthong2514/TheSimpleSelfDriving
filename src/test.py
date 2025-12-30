import cv2
import time
import threading
import numpy as np
import onnxruntime as ort
from uart_protocol import UART
import serial

# ======================
# UART
# ======================
HEADER = 0xAA
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
uart = UART(ser, HEADER)


STATE_LINE_FOLLOW = 0
STATE_TURN_LEFT   = 1
STATE_TURN_RIGHT  = 2
STATE_TURN_AROUND = 3
STATE_STOP        = 4

robot_state = STATE_LINE_FOLLOW

# ======================
# SHARED DATA
# ======================
shared_frame = None          # frame gốc từ camera
shared_yolo = None           # (box, class_id)
shared_line = None           # (cx, cy, contour)
error = 0

sign_id = -1
line_detect_mode = 0

frame_lock = threading.Lock()
running = True

# ======================
# YOLO CONFIG
# ======================
ONNX_MODEL_PATH = "/home/jetson-nano/Desktop/code/Do_an_robot/src/traffic_sign_model.onnx"
INPUT_SIZE = 640
CONF_THRESH = 0.6
IOU_THRESH = 0.45
classes = ['go-ahead', 'stop', 'turn-around', 'turn-left', 'turn-right']

# ======================
# LOAD YOLO
# ======================
providers = ort.get_available_providers()
provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in providers else "CPUExecutionProvider"
session = ort.InferenceSession(ONNX_MODEL_PATH, providers=[provider])
input_name = session.get_inputs()[0].name

# ======================
# PREPROCESS YOLO
# ======================
def preprocess_yolo(img):
    h, w = img.shape[:2]
    scale = INPUT_SIZE / max(h, w)
    nh, nw = int(h * scale), int(w * scale)

    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, np.uint8)

    px, py = (INPUT_SIZE - nw)//2, (INPUT_SIZE - nh)//2
    canvas[py:py+nh, px:px+nw] = resized

    img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img, (2,0,1))[None]
    return img, scale, px, py

# ======================
# NMS
# ======================
def nms(boxes, scores):
    if not boxes:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    x1, y1 = boxes[:,0], boxes[:,1]
    x2, y2 = x1+boxes[:,2], y1+boxes[:,3]
    areas = (x2-x1+1)*(y2-y1+1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2-xx1+1)
        h = np.maximum(0, yy2-yy1+1)
        iou = (w*h)/(areas[i]+areas[order[1:]]-w*h)
        order = order[1:][iou <= IOU_THRESH]

    return keep

# ======================
# LINE DETECTION (NO DRAW)
# ======================
def detect_line(frame):
    h, w = frame.shape[:2]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (0,80,80), (10,255,255)) | \
           cv2.inRange(hsv, (170,80,80), (180,255,255))

    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]

    err = 0
    result = None

    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 500:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                err = cx - w // 2
                result = (cx, cy, c)

    return err, result


# ======================
# CAMERA THREAD
# ======================
def camera_thread(cap):
    global shared_frame
    while running:
        ret, frame = cap.read()
        if not ret:
            continue
        with frame_lock:
            shared_frame = frame

# ======================
# YOLO THREAD (NO DRAW)
# ======================
# ====================== GLOBAL STATE ======================
turn_state = {
    "phase": None,      # None, wait_before_rotate, rotating, wait_after_cross, finished
    "label": None,      # loại biển đang xử lý
    "turn_start_time": None,
    "stop_time": None
}
prev_error = None       # line error trước đó
TURN_WAIT_TIME = 1.5    # giây dừng trước khi quay
STOP_AFTER_CROSS = 1.5  # giây dừng sau khi zero-cross
LINE_CENTER_THRESHOLD = 50  # px gần tâm ảnh để bật lại line

# ====================== YOLO THREAD (CẬP NHẬT LOGIC BIỂN BÁO) ======================
def yolo_thread():
    global sign_id, line_detect_mode, shared_yolo
    global turn_state, prev_error
    last = 0
    while running:
        if time.time() - last < 0.1:
            time.sleep(0.001)
            continue

        with frame_lock:
            if shared_frame is None:
                continue
            frame = shared_frame.copy()

        # ====================== YOLO INFERENCE ======================
        img, scale, px, py = preprocess_yolo(frame)
        preds = session.run(None, {input_name: img})[0][0]

        boxes, scores, ids = [], [], []
        for det in preds:
            conf = det[4]
            if conf < CONF_THRESH:
                continue
            cid = np.argmax(det[5:])
            score = conf * det[5+cid]
            if score < CONF_THRESH:
                continue

            cx,cy,w,h = det[:4]
            x1 = int((cx-w/2-px)/scale)
            y1 = int((cy-h/2-py)/scale)
            boxes.append([x1,y1,int(w/scale),int(h/scale)])
            scores.append(score)
            ids.append(cid)

        keep = nms(boxes, scores)

        # ====================== XỬ LÝ BIỂN BÁO ======================
        if keep:
            i = keep[0]
            sign_id = ids[i]
            shared_yolo = (boxes[i], sign_id)
            label = classes[sign_id]

            if label == 'go-ahead':
                line_detect_mode = 1
                turn_state = {"phase": None, "label": None, "turn_start_time": None, "stop_time": None}
                prev_error = None

            elif label in ['turn-left', 'turn-right', 'turn-around']:
                # Nếu mới thấy biển báo quay và chưa xử lý
                if turn_state["phase"] is None:
                    line_detect_mode = 0
                    turn_state["phase"] = "wait_before_rotate"
                    turn_state["label"] = label
                    turn_state["turn_start_time"] = time.time()
                    prev_error = None

            else:
                line_detect_mode = 0

        else:
            sign_id = -1
            shared_yolo = None

        # ====================== TURN STATE MACHINE ======================
        if turn_state["phase"] == "wait_before_rotate":
            # Chờ tạm trước khi quay
            if time.time() - turn_state["turn_start_time"] >= TURN_WAIT_TIME:
                turn_state["phase"] = "rotating"

        elif turn_state["phase"] == "rotating":
            if shared_line is not None:
                cx, _, _ = shared_line
                w = frame.shape[1]
                err = cx - w//2

                if prev_error is not None:
                    if prev_error * err < 0:  # zero-cross detected
                        turn_state["phase"] = "wait_after_cross"
                        turn_state["stop_time"] = time.time()
                        if turn_state["label"] == 'turn-around':
                            sign_id = classes.index('stop')  # dừng hẳn
                        else:
                            sign_id = -1
                prev_error = err
            # Nếu line mất, giữ prev_error để phát hiện zero-cross

        elif turn_state["phase"] == "wait_after_cross":
            # Dừng sau khi zero-cross
            if time.time() - turn_state["stop_time"] >= STOP_AFTER_CROSS:
                if turn_state["label"] in ['turn-left', 'turn-right']:
                    line_detect_mode = 1  # bật lại line PID
                turn_state["phase"] = "finished"

        last = time.time()



# ======================
# LINE THREAD
# ======================
def line_thread():
    global error, shared_line, robot_state

    while running:
        with frame_lock:
            if shared_frame is None:
                continue
            frame = shared_frame.copy()

        err, result = detect_line(frame)

        if result is None:
            error = 0
            shared_line = None
            continue

        error = err
        shared_line = result

        # ===== KẾT THÚC QUAY =====
        if robot_state in (STATE_TURN_LEFT, STATE_TURN_RIGHT):
            if abs(error) < 10:
                robot_state = STATE_LINE_FOLLOW

        elif robot_state == STATE_TURN_AROUND:
            if abs(error) < 10:
                robot_state = STATE_STOP


# ======================
# DISPLAY THREAD (ONLY DRAW HERE)
# ======================
def display_thread():
    global running
    while running:
        with frame_lock:
            if shared_frame is None:
                continue
            frame = shared_frame.copy()
            yolo = shared_yolo
            line = shared_line

        # Draw YOLO
        if yolo is not None:
            (x,y,w,h), sid = yolo
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
            cv2.putText(frame, classes[sid], (x,y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        # Draw LINE
        if line is not None:
            cx, cy, contour = line
            cv2.circle(frame, (cx, cy), 5, (0,255,0), -1)
            cv2.drawContours(frame, [contour], -1, (0,255,0), 2)


        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == 27:
            running = False
            break

    cv2.destroyAllWindows()

# ======================
# SEND UART THREAD
# ======================
# ====================== SEND THREAD ======================
def send_thread():
    last = 0
    while running:
        if time.time() - last < 0.05:
            continue

        # Gửi đến ESP
        uart.send_uart(line_detect_mode, error, sign_id)
        print(f"MODE:{line_detect_mode} ERR:{error} SIGN:{sign_id}")

        last = time.time()


# ======================
# MAIN
# ======================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

threads = [
    threading.Thread(target=camera_thread, args=(cap,), daemon=True),
    threading.Thread(target=yolo_thread, daemon=True),
    threading.Thread(target=line_thread, daemon=True),
    threading.Thread(target=display_thread, daemon=True),
    threading.Thread(target=send_thread, daemon=True),
]

for t in threads: t.start()
for t in threads: t.join()

cap.release()