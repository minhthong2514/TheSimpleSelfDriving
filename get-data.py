import cv2
import os
import re

# ======== Default folder to save images ========
save_path = r"F:\University\Nam_tu\Do_an_ROBOT\dataset"
os.makedirs(save_path, exist_ok=True)

# ======== Function to get the highest existing image index ========
def get_last_image_index(path):
    max_index = 0
    pattern = re.compile(r"img-(\d+)\.jpg")  # Match files like img-1.jpg
    for file in os.listdir(path):
        match = pattern.match(file)
        if match:
            index = int(match.group(1))
            if index > max_index:
                max_index = index
    return max_index

# Start counting from the next available index
img_count = get_last_image_index(save_path) + 1

# ======== Initialize camera ========
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera!")
        break

    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF

    # Press 'q' to quit
    if key == ord('q'):
        break

    # Capture with SPACE key
    elif key == 32:  # SPACE key code = 32
        filename = os.path.join(save_path, f"img-{img_count}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Captured: {filename}")
        img_count += 1

cap.release()
cv2.destroyAllWindows()
