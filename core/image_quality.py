# core/image_quality.py
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    from piq import brisque
    BRISQUE_AVAILABLE = True
except ImportError:
    BRISQUE_AVAILABLE = False
    logger.warning("PIQ library not installed. BRISQUE metric will not be available.")

def assess_image_quality(image):
    if image is None or image.size == 0:
        return {
            "sharp": False,
            "contrast": False,
            "bright_enough": False,
            "not_noisy": False,
            "brisque": 100 if not BRISQUE_AVAILABLE else None
        }
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    rms_contrast = np.std(gray) / 255.0
    
    brightness = np.mean(gray)
    
    noise_samples = []
    for y in range(0, height, 50):
        for x in range(0, width, 50):
            if y+50 <= height and x+50 <= width:
                patch = gray[y:y+50, x:x+50]
                noise_samples.append(np.var(patch))
    noise = np.mean(noise_samples) if noise_samples else 0
    
    brisque_score = None
    if BRISQUE_AVAILABLE:
        try:
            img_tensor = torch.tensor(image).permute(2,0,1).unsqueeze(0).float() / 255.0
            brisque_score = brisque(img_tensor, data_range=1.0).item()
        except Exception as e:
            logger.error(f"BRISQUE calculation failed: {str(e)}")
            brisque_score = None
    
    return {
        "sharp": laplacian_var > 80,
        "contrast": rms_contrast > 0.3,
        "bright_enough": 60 <= brightness <= 220,
        "not_noisy": noise < 500,
        "brisque": brisque_score
    }