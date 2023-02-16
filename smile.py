import cv2
import datetime

capture = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")
cat_cascade = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")


while True:
    _ , frame = capture.read()
    frameValue = frame.copy()
    gray = cv2.cvtColor(frameValue, cv2.COLOR_BGR2GRAY)

    face = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for x, y, w, h in face:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 255), 2)

        face_roi = frame[y:y+h, x:x+w]
        gray_roi = gray[y:y+h, x:x+w]

        smile = smile_cascade.detectMultiScale(gray_roi, 1.3, 25)

        for x2, y2, w2, h2 in smile:
            cv2.rectangle(face_roi, (x2, y2), (x2+w2, y2+h2), (255, 0 ,255), 2)
            time_stamp = datetime.datetime.now().strftime("%Y-%B-%D-%H-%M-%S")
            file_name = f'selfie-{time_stamp}.png'
            cv2.imwrite(file_name, frameValue)

    cv2.imshow("cam-star", frame)

    if cv2.waitKey(10) == ord("q"):
        break