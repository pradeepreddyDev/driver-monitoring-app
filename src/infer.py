import torch
from torchvision import transforms
from preprocess import detect_face, preprocess_frame
from facenet_pytorch import MTCNN
from config import CONFIG
import cv2
import time

# Load Models
device = "cuda" if torch.cuda.is_available() and CONFIG["device"] == "cuda" else "cpu"
mtcnn = MTCNN(keep_all=True, device=device)
behavior_model = torch.load(CONFIG["behavior_model_path"], map_location=device)
behavior_model.eval()

# Define Transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Video Stream
cap = cv2.VideoCapture(CONFIG["input_video"])
if not cap.isOpened():
    print("Error: Could not open input video.")
    exit()

# Get input frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(CONFIG["output_video"], cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

print("Starting video processing...")
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video stream or error reading the frame.")
        break
    
    # Detect and crop face
    face = detect_face(frame, mtcnn)
    if face is not None:
        try:
            # Preprocess the frame
            input_tensor = transform(preprocess_frame(face)).unsqueeze(0).to(device)

            # Inference
            with torch.no_grad():
                outputs = behavior_model(input_tensor)
                _, predicted = outputs.max(1)

            # Map predictions to behavior
            behavior = {0: "Normal", 1: "Sleeping", 2: "Smoking", 3: "Cellphone"}.get(predicted.item(), "Unknown")
            cv2.putText(frame, behavior, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        except Exception as e:
            print(f"Error during inference: {e}")
    else:
        print("Warning: No face detected in frame.")

    # Write output
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

end_time = time.time()
print(f"Video processing completed in {end_time - start_time:.2f} seconds.")
