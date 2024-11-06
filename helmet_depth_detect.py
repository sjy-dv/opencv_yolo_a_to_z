import cv2 
from ultralytics import YOLO

model = YOLO("./yolov8m.pt")

video_path = "./pixabay_safe_helmet.mp4"

cap = cv2.VideoCapture(video_path)

class_names = model.names
class_ids = {name: idx for idx, name in class_names.items()}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break 

    results = model(frame)

    person_boxes = []
    helmet_boxes = []

    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        conf = result.conf[0]
        class_id = int(result.cls[0])
        label = class_names[class_id]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if label == 'head':
            person_boxes.append((x1, y1, x2, y2))
        elif label == 'helmet':
            helmet_boxes.append((x1, y1, x2, y2))

    for (px1, py1, px2, py2) in person_boxes:
        person_has_helmet = False
        for (hx1, hy1, hx2, hy2) in helmet_boxes:
            if (px1 < hx2 and px2 > hx1 and py1 < hy2 and py2 > hy1):
                person_has_helmet = True
                break  

        if person_has_helmet:
            message = "Ok"
            color = (0, 255, 0)  #green
        else:
            message = "Warning"
            color = (0, 0, 255)  # red

        cv2.putText(frame, message, (px1, py1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


    cv2.imshow('YOLOv8 Result', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


