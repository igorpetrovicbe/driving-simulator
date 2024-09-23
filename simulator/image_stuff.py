import cv2
import numpy as np


def rotate_image(image, angle):
    # Get image dimensions
    (h, w) = image.shape[:2]
    # Calculate the center of the image
    center = (w // 2, h // 2)
    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def overlay_rgba_on_rgb(rgb_img, rgba_img, x, y):

    # Extract alpha channel
    alpha = rgba_img[:, :, 3] / 255.0  # Normalize alpha channel (0 to 1)

    # Resize RGBA image to fit within the RGB image
    h, w = rgba_img.shape[:2]
    roi = rgb_img[y:y + h, x:x + w]

    # Blend RGBA image onto RGB image
    for c in range(0, 3):
        roi[:, :, c] = alpha * rgba_img[:, :, c] + (1 - alpha) * roi[:, :, c]

    # Update RGB image with the blended region
    rgb_img[y:y + h, x:x + w] = roi

    return rgb_img