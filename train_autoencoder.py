# train_autoencoder.py
import os
import cv2
import numpy as np
from core.autoencoder_compressor import AutoencoderCompressor

def load_training_images(path, limit=1000):
    images = []
    for filename in os.listdir(path)[:limit]:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(path, filename)
            img = cv2.imread(img_path)  # Read as color image
            if img is not None:
                img = cv2.resize(img, (256, 256))  # Resize to model's input shape
                img = img.astype(np.float32) / 255.0  # Normalize
                images.append(img)
    return np.array(images)

if __name__ == "__main__":
    train_path = "input/train"

    if not os.path.exists("models"):
        os.makedirs("models")

    compressor = AutoencoderCompressor()
    X_train = load_training_images(train_path)

    if len(X_train) == 0:
        print("No training images found. Please add images to input/train/")
        exit()

    compressor.model.fit(X_train, X_train, epochs=50, batch_size=32, shuffle=True)
    compressor.model.save("models/autoencoder_model.keras")
    print("Training complete. Model saved to models/autoencoder_model.h5")