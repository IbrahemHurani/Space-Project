# main.py
import os
import csv
from glob import glob
import cv2

from core.horizon_detector import detect_horizon
from core.star_detector import detect_stars
from core.image_quality import assess_image_quality
from core.classifier import classify_image
from core.flicker_detector import detect_flickering
from core.compressor import compress_image
#from core.autoencoder_compressor import AutoencoderCompressor  
input_dir = "input/"
output_dir = "output/processed/"
summary_path = "summary.csv"

def classify_and_process_batch(input_dir, output_dir, summary_path):
    os.makedirs(output_dir, exist_ok=True)

    
    #compressor = AutoencoderCompressor(model_path="models/autoencoder_model.keras")
    files = glob(os.path.join(input_dir, "**", "*.*"), recursive=True)
    files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    files.sort()

    if not files:
        print("No images found.")
        return

    prev_image = None

    with open(summary_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "label", "has_horizon", "has_stars", "sharp", "bright_enough", "not_noisy", "flickering"])

        for file_path in files:
            image = cv2.imread(file_path)
            if image is None:
                print(f"Could not load image: {file_path}")
                continue

            filename = os.path.basename(file_path)
            subfolder = os.path.basename(os.path.dirname(file_path)).lower()

            # Horizon detection
            if "star" not in subfolder:
                horizon_y, horizon_img = detect_horizon(image)
                has_horizon = horizon_y is not None
                if horizon_img is not None:
                    debug_path = os.path.join("output", f"horizon_{filename}")
                    cv2.imwrite(debug_path, horizon_img)
            else:
                has_horizon = False

            # Star detection
            if "hz_horizon" not in subfolder:
                has_stars, star_img = detect_stars(image)
                if star_img is not None:
                    debug_path = os.path.join("output", f"stars_{filename}")
                    cv2.imwrite(debug_path, star_img)
            else:
                has_stars = False

            # Quality and classification
            quality = assess_image_quality(image)
            label = classify_image(has_horizon, has_stars, quality)

            # Flicker detection
            flickering = False
            if prev_image is not None:
                flickering = detect_flickering(prev_image, image)
            prev_image = image

            output_name = f"{label}_{filename}"
            output_path = os.path.join(output_dir, output_name)

            # שימוש בדחסן (JPEG / Autoencoder)
            compress_image(image, output_path)
            #compressed_img = compressor.compress(image)
            #cv2.imwrite(output_path, compressed_img)

            writer.writerow([filename, label, has_horizon, has_stars,
                             quality["sharp"], quality["bright_enough"], quality["not_noisy"], flickering])
            print(f"{filename}: classified as {label}, Flickering: {flickering}")

if __name__ == "__main__":
    classify_and_process_batch(input_dir, output_dir, summary_path)
