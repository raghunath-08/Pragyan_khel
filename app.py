from flask import Flask, render_template, Response, request, jsonify
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO model
model = YOLO("yolov8n.pt")


cap = cv2.VideoCapture(0)

# Global variables
selected_box = None
boxes = []


@app.route('/select', methods=['POST'])
def select():
    global selected_box, boxes

    data = request.json
    click_x = data['x']
    click_y = data['y']

    for (x1, y1, x2, y2) in boxes:
        if x1 < click_x < x2 and y1 < click_y < y2:
            selected_box = (x1, y1, x2, y2)
            return jsonify({"status": "selected"})

    return jsonify({"status": "not_found"})



def generate_frames():
    global boxes, selected_box

    while True:
        ret, frame = cap.read()
        if not ret:
            break

       
        frame = cv2.flip(frame, 1)

        
        results = model(frame, imgsz=320, conf=0.5)

        boxes = []

        for r in results:
            for box in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                boxes.append((x1, y1, x2, y2))

        
        mask = None

        if selected_box:
            cx = (selected_box[0] + selected_box[2]) // 2
            cy = (selected_box[1] + selected_box[3]) // 2

            for (x1, y1, x2, y2) in boxes:
                if x1 < cx < x2 and y1 < cy < y2:
                    mask = (x1, y1, x2, y2)
                    break

        # Apply blur
        if mask:
            x1, y1, x2, y2 = mask

            # Blur full frame
            blurred = cv2.GaussianBlur(frame, (35, 35), 0)

            # Keep selected area sharp
            blurred[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

            frame = blurred

            # Draw green box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Convert to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == "__main__":
    app.run(debug=True)
