import cv2
import numpy as np

def detect_stars(image):
    # המרה לגווני אפור
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # שיפור ניגודיות עם CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # סף אדפטיבי כדי לזהות גם כוכבים חלשים ברקע משתנה
    thresh = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=-10
    )

    # ניקוי רעשים קטנים
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=1)

    # מציאת קונטורים
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # העתקת התמונה לסימון
    marked_image = image.copy()
    stars_count = 0

    for c in contours:
        area = cv2.contourArea(c)
        if 6 < area < 150:  # סינון משופר
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w) / h
            if 0.6 < aspect_ratio < 1.5:  # יחס גובה-רוחב גמיש יותר
                padding = 3  # ריבוע ברור אבל לא מוגזם
                top_left = (x - padding, y - padding)
                bottom_right = (x + w + padding, y + h + padding)
                cv2.rectangle(marked_image, top_left, bottom_right, (0, 255, 0), 2)
                stars_count += 1

    circles = cv2.HoughCircles(enhanced, cv2.HOUGH_GRADIENT, dp=1.2, minDist=5,
                               param1=50, param2=15, minRadius=1, maxRadius=5)

    if circles is not None:
        for circle in circles[0, :]:
            center = (int(circle[0]), int(circle[1]))
            cv2.circle(marked_image, center, 3, (255, 0, 0), 1)
        stars_count += circles.shape[1]

    return (stars_count >= 3), marked_image
