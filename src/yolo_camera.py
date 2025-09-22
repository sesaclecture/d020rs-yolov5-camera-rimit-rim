import torch
import cv2

# Model​
model = torch.hub.load("ultralytics/yolov5", "yolov5m")

# Video capture
cap = cv2.VideoCapture(0)

# TODO: Loop for camera frames
while True:
    # Read frame (BGR to RGB)
    ret, frame = cap.read()
    # TODO: break the loop on error
    if not ret:
        break

    # 추론 실행 (BGR -> RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)

    # TODO: Boudning box 그리기
    for i, obj in enumerate(results.xyxy[0]):
        # TODO: 인식결과를 표시하기 위한 좌표를 얻음
        x1, y1, x2, y2 = map(int, obj[:4])
        
        # TODO: 인식된 정확도(confidence)와 클래스를 label로 구성
        conf = float(obj[4])        # confidence
        cls = int(obj[5])        # 클래스 id

        # TODO: OpenCV를 이용해서 해당 좌표에 사각형과 text를 출력
        obj_info = list(map(int, obj))
        # print(f"Object {i}: {model.names[obj_info[5]]}")
        cv2.rectangle(frame, (x1, y1), (x1 + x2, y1 + y2), (0, 255, 0), 2)
        cv2.putText(frame, model.names[cls], (x1 + 40, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # TODO: 화면 표시
    cv2.imshow("yolo_camera", frame)

    # TODO: 종료를 위한 key 처리
    if cv2.waitKey(1) & 0xff == 27:     # 27: ESC
        break

cap.release()
cv2.destroyAllWindows()
