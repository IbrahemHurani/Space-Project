def classify_image(has_horizon, has_stars, quality):
    # הגדרות ספי איכות
    sharp_threshold = 80
    contrast_threshold = 0.3
    noise_threshold = 500
    brisque_threshold = 40
    
    # תמונת אופק טובה
    if (has_horizon and 
        quality["sharp"] and 
        quality["contrast"] and
        quality["bright_enough"]):
        return "good_horizon"
    
    # תמונת כוכבים
    if (has_stars and 
        quality["not_noisy"] and 
        quality["bright_enough"] and
        (quality["brisque"] is None or quality["brisque"] < brisque_threshold)):
        return "star_view"
    
    # תמונה כללית טובה
    if (quality["sharp"] and 
        quality["bright_enough"] and 
        quality["contrast"] and
        quality["not_noisy"]):
        return "perfect"
    
    # תמונה מקובלת
    if (quality["brisque"] is not None and 
        quality["brisque"] < brisque_threshold and
        quality["sharp"] and
        quality["bright_enough"]):
        return "acceptable"
    
    return "rejected"