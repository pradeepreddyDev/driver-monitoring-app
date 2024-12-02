import cv2
from facenet_pytorch import MTCNN


def detect_face(frame, mtcnn):
    """
    Detect face in the given frame using MTCNN.
    Args:
        frame: The input image frame.
        mtcnn: The MTCNN object for face detection.

    Returns:
        Cropped face image if detected, else None.
    """
    # Detect faces in the frame
    boxes, _ = mtcnn.detect(frame)
    if boxes is not None:
        # Select the largest face, assuming it's the driver
        largest_face = max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
        x1, y1, x2, y2 = map(int, largest_face)
        cropped_face = frame[y1:y2, x1:x2]
        return cropped_face
    return None


def preprocess_frame(frame):
    """
    Preprocess the frame for behavior classification.
    Args:
        frame: The input image frame.

    Returns:
        Preprocessed image ready for input to a model.
    """
    # Resize the frame to match the model input requirements (224x224) and normalize pixel values
    frame_resized = cv2.resize(frame, (224, 224))

    # Normalize the pixel values (convert from 0-255 to 0-1 range)
    frame_normalized = frame_resized / 255.0

    return frame_normalized
