import cv2
import numpy as np

def detect_stars(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, thresh = cv2.threshold(enhanced, 220, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    marked_image = image.copy()
    stars_count = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area < 30:
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if 0.6 < circularity <= 1.2:
                (x, y), radius = cv2.minEnclosingCircle(c)
                center = (int(x), int(y))
                cv2.circle(marked_image, center, 3, (255, 0, 0), 1)
                stars_count += 1

    # חיפוש עיגולים קטנים בעזרת HoughCircles
    circles = cv2.HoughCircles(enhanced, cv2.HOUGH_GRADIENT, dp=1.2, minDist=5,
                               param1=50, param2=15, minRadius=1, maxRadius=5)

    if circles is not None:
        for circle in circles[0, :]:
            center = (int(circle[0]), int(circle[1]))
            cv2.circle(marked_image, center, 3, (255, 0, 0), 1)
        stars_count += circles.shape[1]

    return (stars_count >= 3), marked_image
