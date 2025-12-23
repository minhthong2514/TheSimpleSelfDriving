import onnxruntime as ort
import numpy as np
import cv2
import time

# ======= Cấu hình =======
ONNX_MODEL_PATH = "traffic_sign_model.onnx"  # đường dẫn ONNX
IMG_PATH = "/home/jetson-nano/Desktop/code/Do_an_robot/test_data/go_ahead1.jpeg"  # ảnh cần detect
INPUT_SIZE = 640
CONF_THRESH = 0.25
IOU_THRESH = 0.45
classes = ['go-ahead', 'stop', 'turn-around', 'turn-left', 'turn-right']  # tên class

# ======= Load ONNX model =======
# providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] nếu có GPU
session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CUDAExecutionProvider'])
input_name = session.get_inputs()[0].name

# ======= Hàm tiền xử lý =======
def preprocess(img):
    h, w = img.shape[:2]
    scale = INPUT_SIZE / max(h, w)
    nh, nw = int(h*scale), int(w*scale)                                                                                                                                                                             
    # resize
    img_resized = cv2.resize(img, (nw, nh))
    # padding
    pad_x = (INPUT_SIZE - nw) // 2
    pad_y = (INPUT_SIZE - nh) // 2
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y+nh, pad_x:pad_x+nw] = img_resized
    # BGR -> RGB, normalize
    img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # HWC -> CHW
    img_rgb = np.transpose(img_rgb, (2,0,1))
    img_rgb = np.expand_dims(img_rgb, axis=0)
    return img_rgb, scale, pad_x, pad_y, nw, nh

# ======= NMS =======
def nms(boxes, scores, iou_thresh):
    if len(boxes) == 0:
        return []
    boxes_np = np.array(boxes).tolist()  # cv2.dnn.NMSBoxes yêu cầu list of [x,y,w,h]
    scores_np = np.array(scores).tolist()
    idxs = cv2.dnn.NMSBoxes(boxes_np, scores_np, CONF_THRESH, iou_thresh)
    return idxs.flatten() if len(idxs) > 0 else []

# ======= Load ảnh =======
img = cv2.imread(IMG_PATH)
if img is None:
    raise ValueError(f"Không load được ảnh {IMG_PATH}")

img_input, scale, pad_x, pad_y, nw, nh = preprocess(img)

# ======= Infer + đo thời gian =======
start_time = time.time()
outputs = session.run(None, {input_name: img_input})
end_time = time.time()
pred = outputs[0]  # [1, num_boxes, 85] (x,y,w,h,conf,cls_probs)
print(f"ONNX Inference time: {(end_time - start_time)*1000:.2f} ms")
print(f"Detected {len(pred[0])} boxes before NMS")

# ======= Xử lý kết quả =======
boxes, scores, class_ids = [], [], []

for det in pred[0]:
    conf = det[4]
    if conf < CONF_THRESH:
        continue
    class_prob = det[5:]
    class_id = int(np.argmax(class_prob))
    score = conf * class_prob[class_id]
    if score < CONF_THRESH:
        continue

    cx, cy, w, h = det[0:4]

    # Chuyển về tọa độ ảnh gốc
    x1 = int((cx - w/2 - pad_x) / nw * img.shape[1])
    y1 = int((cy - h/2 - pad_y) / nh * img.shape[0])
    x2 = int((cx + w/2 - pad_x) / nw * img.shape[1])
    y2 = int((cy + h/2 - pad_y) / nh * img.shape[0])

    boxes.append([x1, y1, x2-x1, y2-y1])
    scores.append(float(score))
    class_ids.append(class_id)

# ======= NMS =======
idxs = nms(boxes, scores, IOU_THRESH)
print(f"Boxes after NMS: {len(idxs)}")

# ======= Vẽ kết quả =======
for i in idxs:
    x, y, w, h = boxes[i]
    label = classes[class_ids[i]] if class_ids[i] < len(classes) else str(class_ids[i])
    score = scores[i]
    print(f"Box: {x},{y},{w},{h} | Label: {label} | Score: {score:.2f}")
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.putText(img, f"{label}:{score:.2f}", (x,y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

# ======= Hiển thị ảnh =======
cv2.imshow("ONNX YOLO Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
