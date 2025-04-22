import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
# MediaPipe-Parameter für bessere Erkennung anpassen
hands = mpHands.Hands(min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True: 
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                # Alle Landmarkpunkte einzeichnen (mit verschiedenen Farben)
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                
                # Daumenspitze besonders hervorheben (ID 4)
                if id == 4:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                    cv2.putText(img, f"Daumen", (cx-20, cy-15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
            
            # Hand-Verbindungen zeichnen (nach den Kreisen, damit die Kreise sichtbar bleiben)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # FPS-Berechnung
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    # FPS anzeigen
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    # Mit ESC beenden können
    key = cv2.waitKey(1)
    if key == 27:  # ESC-Taste
        break

# Kamera freigeben und Fenster schließen
cap.release()
cv2.destroyAllWindows()

#cd /Users/konstantin/Desktop/Advanced_Computer_Vision && source HandTrackingProject/venv/bin/activate && python handTrackingMin.py
