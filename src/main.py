import cv2
import time
import threading
import numpy as np
import onnxruntime as ort
from uart_protocol import UART
import serial


class TurnScript:
    def __init__(self, turn_mode, final_stop):
        self.turn_mode = turn_mode
        self.final_stop = final_stop

        self.step = 0
        self.start_time = None
        self.active = True

        self.line_lost = False       # đã mất line chưa
        self.line_found_time = None # thời điểm tìm lại line

    def update(self, line_detected, line_error):
        global mode

        # STEP 0: STOP trước khi quay
        if self.step == 0:
            mode = MODE_STOP
            if self.start_time is None:
                self.start_time = time.time()
            elif time.time() - self.start_time >= 3:
                self.step = 1
                self.start_time = None

        # STEP 1: QUAY (KHÔNG PHỤ THUỘC LINE)
        elif self.step == 1:
            mode = self.turn_mode

            # Nếu line biến mất -> đánh dấu
            if not line_detected:
                self.line_lost = True

            # Chỉ xét dừng khi:
            # - đã từng mất line
            # - và line xuất hiện lại
            if self.line_lost and line_detected:
                # yêu cầu line ổn định 0.2s để tránh nhiễu
                if self.line_found_time is None:
                    self.line_found_time = time.time()
                elif time.time() - self.line_found_time > 0.2:
                    self.step = 2
                    self.start_time = None
            else:
                self.line_found_time = None

        # STEP 2: STOP sau khi quay
        elif self.step == 2:
            mode = MODE_STOP
            if self.start_time is None:
                self.start_time = time.time()
            elif time.time() - self.start_time >= 3:
                if self.final_stop:
                    self.active = False   # turn-around → đứng luôn
                else:
                    mode = MODE_TURN_AHEAD
                    self.active = False


current_script = None

# ======================
# UART
# ======================
HEADER = 0xAA
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
uart = UART(ser, HEADER)


MODE_TURN_AHEAD = 0   # bám line
MODE_STOP       = 1
MODE_TURN_AROUND= 2
MODE_TURN_LEFT  = 3
MODE_TURN_RIGHT = 4


mode = MODE_STOP      # mặc định khi bật nguồn
sign_id = -1          # chỉ để debug

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
CONF_THRESH = 0.8
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
TURN_WAIT_TIME = 2    # giây dừng trước khi quay
STOP_AFTER_CROSS = 2  # giây dừng sau khi zero-cross
LINE_CENTER_THRESHOLD = 100  # px gần tâm ảnh để bật lại line

# ====================== YOLO THREAD (CẬP NHẬT LOGIC BIỂN BÁO) ======================
def yolo_thread():
    global sign_id, shared_yolo, current_script, mode

    last = 0
    while running:
        if time.time() - last < 0.1:
            continue

        with frame_lock:
            if shared_frame is None:
                continue
            frame = shared_frame.copy()

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

        if keep and current_script is None:
            i = keep[0]
            sign_id = ids[i]
            shared_yolo = (boxes[i], sign_id)

            if sign_id == 0:   # go-ahead
                mode = MODE_TURN_AHEAD

            elif sign_id == 1: # stop
                mode = MODE_STOP

            elif sign_id == 2: # turn around
                current_script = TurnScript(MODE_TURN_AROUND, final_stop=True)

            elif sign_id == 3: # turn left
                current_script = TurnScript(MODE_TURN_LEFT, final_stop=False)

            elif sign_id == 4: # turn right
                current_script = TurnScript(MODE_TURN_RIGHT, final_stop=False)

        else:
            shared_yolo = None
            sign_id = -1

        last = time.time()




# ======================
# LINE THREAD
# ======================
def line_thread():
    global error, shared_line, current_script

    while running:
        with frame_lock:
            if shared_frame is None:
                continue
            frame = shared_frame.copy()

        err, result = detect_line(frame)

        if result:
            error = err
            shared_line = result
        else:
            error = 0
            shared_line = None

        if current_script:
            line_detected = shared_line is not None
            current_script.update(line_detected, error)
            if not current_script.active:
                current_script = None




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

        uart.send_uart(mode, error)
        print(f"MODE:{mode} ERR:{error} SIGN:{classes[sign_id] if sign_id!=-1 else 'none'}")

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