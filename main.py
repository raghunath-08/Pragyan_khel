import cv2
from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

selected_box = None

# Mouse click function
def select_object(event, x, y, flags, param):
    global selected_box
    if event == cv2.EVENT_LBUTTONDOWN:
        for box in param:
            x1, y1, x2, y2 = map(int, box)
            if x1 < x < x2 and y1 < y < y2:
                selected_box = (x1, y1, x2, y2)

cv2.namedWindow("AI Tracking")
cv2.setMouseCallback("AI Tracking", select_object)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Fix mirror
    frame = cv2.flip(frame, 1)

    frame = cv2.resize(frame, (640, 480))

    # Detect objects
    results = model(frame, imgsz=320, conf=0.5, verbose=False)

    boxes_list = []

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        for box in boxes:
            boxes_list.append(box)

    # Blur background
    blurred = cv2.GaussianBlur(frame, (25, 25), 0)

    output = blurred.copy()

    # If object selected → keep it clear
    if selected_box is not None:
        x1, y1, x2, y2 = selected_box
        output[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
        cv2.rectangle(output, (x1, y1), (x2, y2), (0,255,0), 2)

    # Show all boxes (optional)
    for box in boxes_list:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(output, (x1, y1), (x2, y2), (255,0,0), 1)

    cv2.imshow("AI Tracking", output)

    # Update mouse param
    cv2.setMouseCallback("AI Tracking", select_object, boxes_list)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows() 
