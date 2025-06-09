import cv2
import numpy as np
import torch


def detect_horizon(image):
   
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    largest_contour = max(contours, key=cv2.contourArea)

    points = np.squeeze(largest_contour)
    if points.ndim != 2:
        return None, None

    xs = points[:, 0]
    ys = points[:, 1]

    coeffs = np.polyfit(xs, ys, 2)
    poly = np.poly1d(coeffs)

    marked_image = image.copy()
    for x in range(0, image.shape[1], 2):
        y = int(poly(x))
        if 0 <= y < image.shape[0]:
            cv2.circle(marked_image, (x, y), 1, (0, 255, 0), -1)

    avg_y = int(np.mean(ys))
    return avg_y, marked_image
