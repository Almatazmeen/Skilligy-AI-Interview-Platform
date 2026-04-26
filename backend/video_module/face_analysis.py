# ============================================================
# Face Analysis — Landmark Detection (No Emotion Inference)
# ============================================================

import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# Initialize once (important for performance)
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

def analyze_frame(frame):
    """
    Analyze a single video frame.
    
    Returns:
        {
          "face_detected": bool,
          "landmarks": [(x, y), ...] or []
        }
    """

    if frame is None:
        return {
            "face_detected": False,
            "landmarks": []
        }

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return {
            "face_detected": False,
            "landmarks": []
        }

    h, w, _ = frame.shape
    landmarks = []

    for lm in result.multi_face_landmarks[0].landmark:
        x = int(lm.x * w)
        y = int(lm.y * h)
        landmarks.append((x, y))

    return {
        "face_detected": True,
        "landmarks": landmarks
    }
