import cv2
import numpy as np

def detect_flickering(prev_img, current_img, threshold=0.25, min_change=0.05):
    if prev_img is None or current_img is None:
        return False
    if prev_img.shape != current_img.shape:
        return False
        
    prev_yuv = cv2.cvtColor(prev_img, cv2.COLOR_BGR2YUV)
    curr_yuv = cv2.cvtColor(current_img, cv2.COLOR_BGR2YUV)
    
    prev_y = prev_yuv[:,:,0].astype(np.float32)
    curr_y = curr_yuv[:,:,0].astype(np.float32)
    
    diff = np.abs(prev_y - curr_y)
    normalized_diff = diff / np.maximum(prev_y, 1) 
    
    mean_diff = np.mean(normalized_diff)
    if mean_diff < min_change:
        return False
    
    changed_pixels = np.sum(normalized_diff > threshold)
    total_pixels = prev_y.size
    ratio = changed_pixels / total_pixels
    
    return ratio > 0.15
