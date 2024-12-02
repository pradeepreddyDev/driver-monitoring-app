import os
import torch
import torchvision.models as models
from torchvision import transforms
from facenet_pytorch import MTCNN
from config import CONFIG
import cv2
import time
from datetime import datetime

# Load Models
device = "cuda" if torch.cuda.is_available() and CONFIG["device"] == "cuda" else "cpu"

# Initialize MTCNN for face detection with adjusted thresholds
mtcnn = MTCNN(
    keep_all=True, device=device, min_face_size=20, thresholds=[0.6, 0.7, 0.7]
)

# Load ResNet18 Behavior Model using state_dict
behavior_model = models.resnet18()  # Initialize model architecture
state_dict = torch.load(
    CONFIG["behavior_model_path"], map_location=device
)  # Load model weights
behavior_model.load_state_dict(state_dict)  # Load state_dict into the model
behavior_model.eval()  # Set model to evaluation mode
behavior_model.to(device)  # Ensure model is on the correct device

# Define Transformation
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Ensure resizing for input consistency
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Define detect_face function using MTCNN
def detect_face(frame, mtcnn):
    """
    Detects a face in the given frame using MTCNN and returns the cropped face.
    If no face is found, returns None.
    """
    boxes, _ = mtcnn.detect(frame)
    if boxes is not None:
        box = boxes[0]
        x1, y1, x2, y2 = [int(coord) for coord in box]
        face = frame[y1:y2, x1:x2]
        return face
    return None


def process_frame(frame, mtcnn, behavior_model, transform, output_images_path):
    """
    Processes a single video frame: detects faces, predicts behavior, and optionally saves images.
    """
    preprocessed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    preprocessed_frame = cv2.equalizeHist(preprocessed_frame)
    preprocessed_frame = cv2.cvtColor(preprocessed_frame, cv2.COLOR_GRAY2BGR)

    # Detect faces using MTCNN
    boxes, _ = mtcnn.detect(preprocessed_frame)

    if boxes is not None:
        print(f"Detected {len(boxes)} face(s).")
        for box in boxes:
            x1, y1, x2, y2 = [int(coord) for coord in box]
            face = frame[y1:y2, x1:x2]

            if face is not None:
                try:
                    # Resize and preprocess face
                    face_resized = cv2.resize(face, (224, 224))
                    input_tensor = (
                        transform(face_resized).unsqueeze(0).to(device).float()
                    )

                    # Perform inference
                    with torch.no_grad():
                        outputs = behavior_model(input_tensor)
                        probabilities = torch.softmax(outputs, dim=1)
                        print(f"Model outputs: {outputs}")
                        print(f"Probabilities: {probabilities}")
                        _, predicted = outputs.max(1)

                    # Map prediction to behavior
                    behavior = {
                        0: "Normal",
                        1: "Sleeping",
                        2: "Smoking",
                        3: "Cellphone",
                    }.get(predicted.item(), "Unknown")
                    print(f"Predicted behavior: {behavior}")

                    # Draw bounding box
                    color = (0, 255, 0) if behavior == "Normal" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        behavior,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        color,
                        2,
                    )

                    # Save frame if violation
                    if behavior in ["Sleeping", "Smoking", "Cellphone"]:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_image_filename = f"{behavior}_{timestamp}.jpg"
                        output_image_path = os.path.join(
                            output_images_path, output_image_filename
                        )
                        cv2.imwrite(output_image_path, frame)
                        print(f"Saved violation image: {output_image_path}")

                except Exception as e:
                    print(f"Error during inference: {e}")
    else:
        print("No faces detected.")
    return frame


# Main Video Stream Processing
def main():
    # Initialize video capture
    test_behavior_model(
        "/home/pradeep/november/driver-monitoring-app/src/Photo-1.jpeg",
        behavior_model,
        transform,
    )
    return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam. Please check if the webcam is connected.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output video writer
    out = None
    if CONFIG.get("output_video"):
        out = cv2.VideoWriter(
            CONFIG["output_video"],
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            (frame_width, frame_height),
        )

    # Create output directory for images
    output_images_path = "./output_images"
    os.makedirs(output_images_path, exist_ok=True)

    print("Starting webcam video stream processing...")
    start_time = time.time()

    frame_skip_interval = 5
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            frame_count += 1
            if not ret or frame is None or not frame.any():
                print("Invalid or empty frame detected, skipping.")
                continue
            if frame_count % frame_skip_interval != 0:
                continue

            # Process frame
            processed_frame = process_frame(
                frame, mtcnn, behavior_model, transform, output_images_path
            )

            # Display frame
            cv2.imshow("Driver Monitoring", processed_frame)

            # Write to video file if enabled
            if out is not None:
                out.write(processed_frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        end_time = time.time()
        print(
            f"Webcam stream processing completed in {end_time - start_time:.2f} seconds."
        )


from PIL import Image


def test_behavior_model(image_path, behavior_model, transform):
    """
    Tests the behavior model with a static image.

    Args:
        image_path (str): Path to the test image.
        behavior_model (torch.nn.Module): The behavior model.
        transform (torchvision.transforms.Compose): Transformation pipeline for preprocessing.

    Returns:
        None
    """
    try:
        print(f"Testing behavior model with image: {image_path}")

        # Load the image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            print(
                f"Error: Unable to load image at {image_path}. Please verify the path."
            )
            return

        # Convert the image from BGR to RGB (model expects RGB input)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert the NumPy array to a PIL image
        image_pil = Image.fromarray(image_rgb)

        # Apply transformations to prepare the image as model input
        input_tensor = transform(image_pil).unsqueeze(0).to(device).float()

        # Run inference with the behavior model
        with torch.no_grad():
            outputs = behavior_model(input_tensor)
            probabilities = torch.softmax(
                outputs, dim=1
            )  # Convert logits to probabilities
            _, predicted = outputs.max(1)  # Get the index of the highest probability

        # Map the predicted class index to behavior
        behavior_mapping = {0: "Normal", 1: "Sleeping", 2: "Smoking", 3: "Cellphone"}
        predicted_behavior = behavior_mapping.get(predicted.item(), "Unknown")

        # Print and log the results
        print("----- Test Results -----")
        print(f"Model Outputs (Logits): {outputs.cpu().numpy()}")
        print(f"Predicted Probabilities: {probabilities.cpu().numpy()}")
        print(f"Predicted Behavior: {predicted_behavior}")
        print("------------------------")

        # Display the image with the predicted label
        cv2.putText(
            image,
            f"Predicted: {predicted_behavior}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )
        cv2.imshow("Test Image - Predicted Behavior", image)
        cv2.waitKey(0)  # Wait for a key press to close the window
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error during test_behavior_model: {e}")


if __name__ == "__main__":
    main()
