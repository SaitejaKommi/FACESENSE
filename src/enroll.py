import os
import json
from deepface import DeepFace

ENROLL_DIR = "data/enrollments"
OUTPUT_FILE = "data/embeddings.json"

def generate_embeddings():
    all_embeddings = []

    for person in os.listdir(ENROLL_DIR):
        person_dir = os.path.join(ENROLL_DIR, person)

        if not os.path.isdir(person_dir):
            continue

        print(f"\nProcessing: {person}")

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)

            try:
                embedding = DeepFace.represent(
                    img_path = img_path,
                    model_name = "Facenet512",
                    detector_backend = "retinaface"
                )

                all_embeddings.append({
                    "name": person,
                    "image": img_path,
                    "embedding": embedding[0]["embedding"]
                })

                print(f"  OK → {img_name}")

            except Exception as e:
                print(f"  Failed: {img_name}, error: {e}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_embeddings, f, indent=4)

    print(f"\n✔ Embeddings saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_embeddings()
