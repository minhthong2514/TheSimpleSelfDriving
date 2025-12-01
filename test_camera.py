import cv2

# Open the default camera
cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()

    # Display the captured frame
    cv2.imshow('Camera', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cap.release()
cv2.destroyAllWindows()