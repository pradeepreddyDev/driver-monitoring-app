CONFIG = {
    "input_video": "rtsp://admin:deepu27017@192.168.29.4/stream0",  # Your RTSP stream URL here
    "output_video": "output.mp4",
    "device": "cuda",
    "behavior_model_path": "./models/behavior_detection/resnet18.pth",
    "face_model_path": "./models/face_detection/mtcnn",
}
