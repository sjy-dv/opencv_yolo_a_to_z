import cv2
from ultralytics import YOLO

model = YOLO("./runs/detect/train2/weights/best.pt")

video_path = "./pixabay_safe_helmet.mp4"

cap = cv2.VideoCapture(video_path)


while cap.isOpened():
    ret, frame = cap.read()
    #end frame
    if not ret:
        break 
    
    results = model(frame)
    
    for result in results[0].boxes:
        print(result)
        x1,y1,x2,y2 = map(int, result.xyxy[0])
        conf = result.conf[0]
        class_id = int(result.cls[0])
        label = model.names[class_id]
        
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        
    cv2.imshow('YOLOv8 Result', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()