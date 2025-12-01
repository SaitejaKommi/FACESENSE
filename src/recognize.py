import cv2
from deepface import DeepFace
import json
import numpy as np
import os
from datetime import datetime
from blink_detector import detect_blinks
from spoof_detector import is_spoof

# Path to files
EMBEDDINGS_FILE = "data/embeddings.json"
ATTENDANCE_FILE = "data/attendance.csv"

# Load embeddings
with open(EMBEDDINGS_FILE, "r") as f:
    embeddings = json.load(f)

def cosine_similarity(e1, e2):
    e1, e2 = np.array(e1), np.array(e2)
    return np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))

def recognize_face(frame):
    try:
        result = DeepFace.represent(frame, model_name="Facenet512", enforce_detection=True)
        emb = result[0]["embedding"]
    except:
        return "Unknown", 0

    best_name = "Unknown"
    best_sim = 0

    for person in embeddings:
        sim = cosine_similarity(emb, person["embedding"])
        if sim > best_sim:
            best_sim = sim
            best_name = person["name"]

    if best_sim < 0.65:
        return "Unknown", best_sim

    return best_name, best_sim

def mark_attendance(name):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{name},{now},Present\n"

    # Avoid double entry
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "r") as f:
            for row in f.readlines():
                if name in row and row.startswith(name) and row.split(",")[1][:10] == now[:10]:
                    print("Already marked today.")
                    return

    with open(ATTENDANCE_FILE, "a") as f:
        f.write(line)

    print("Attendance marked for", name)


# ---------- MAIN LOGIC ----------
print("Starting Live Liveness + Recognition...")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    frame = cv2.resize(frame, (640, 480))

    # 1️⃣ Anti-Spoof
    if is_spoof(frame):
        cv2.putText(frame, "SPOOF DETECTED!", (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("FaceSense", frame)
        continue

    # 2️⃣ Blink detection
    if detect_blinks(frame):
        cv2.putText(frame, "Blink Verified", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # 3️⃣ Recognition
        name, sim = recognize_face(frame)
        cv2.putText(frame, f"{name} ({sim:.2f})", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if name != "Unknown":
            mark_attendance(name)
            break

    cv2.imshow("FaceSense", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
