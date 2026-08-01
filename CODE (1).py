import cv2

# Initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Start video capture (0 = default webcam)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Lower resolution for speed
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster processing
    frame_resized = cv2.resize(frame, (640, 480))

    # Detect people in the image
    # returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(frame_resized, winStride=(8,8), padding=(16,16), scale=1.05)

    # Draw rectangles with colored lines (green)
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame_resized, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green color, thickness 2

    # Show the output frame
    cv2.imshow("Full Body Detection", frame_resized)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()





