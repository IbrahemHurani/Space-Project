import cv2

def compress_image(image, output_path, method="jpeg", quality=90):
    if method == "jpeg":
        cv2.imwrite(output_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    elif method == "jpeg2000":
        cv2.imwrite(output_path, image, [int(cv2.IMWRITE_JPEG2000_COMPRESSION_X1000), 500])
