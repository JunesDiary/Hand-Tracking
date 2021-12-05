import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mphands = mp.solutions.hands
hands = mphands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLine in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLine, mphands.HAND_CONNECTIONS)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
