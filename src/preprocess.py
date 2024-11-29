import cv2
from facenet_pytorch import MTCNN

def detect_face(frame, mtcnn):
    """
    Detect face in the given frame using MTCNN.
    """
    boxes, _ = mtcnn.detect(frame)
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            return frame[y1:y2, x1:x2]
    return None

def preprocess_frame(frame):
    """
    Preprocess the frame for behavior classification.
    """
    # Resize, normalize, or any additional preprocessing
    return cv2.resize(frame, (224, 224)) / 255.0
