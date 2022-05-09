import cv2
import os

if __name__ == "__main__":
    # почему-то если путь к конфигу не абсолютный, то его не воспринимают и это все не работает
    cascPath = "C:\\CompositePortraitRecongnition\\configs\\cascade.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    # video_capture = cv2.VideoCapture(0)
    video_capture = cv2.VideoCapture('C:\\CompositePortraitRecongnition\\video\\test.mp4')
    count = 0
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(gray, 1.3, 5)

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imwrite("dataset/frame%d.jpg" % count, gray[y:y + h, x:x + w])
            count += 1
            # здесь надо добавить сравнение с эмбеддингами из базы данных

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # пока оставлю тут всего 5 обнаруженных фрагментов
        if count >= 5:
            break

        # чтобы прервать поиск, нажмите на q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
