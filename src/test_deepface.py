from deepface import DeepFace

print(
    DeepFace.analyze(
        img_path=r"data/raw_frames/frame_saved.jpg",
        actions=['age', 'gender', 'emotion'],
        detector_backend='retinaface',
        enforce_detection=False
    )
)
