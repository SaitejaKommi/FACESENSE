from deepface import DeepFace

result = DeepFace.analyze(
    img_path="data/raw_frames/frame.jpg",
    actions=['age', 'gender', 'emotion']
)

print(result)
