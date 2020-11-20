# Modules
import os
import cv2
import random
import numpy as np

# file number counter
counter = 1


def returnFileName(temp):
    return "./Generated/Generated-" + str(temp) + ".png"


def brighten(image):
    new_image = np.zeros(image.shape, image.dtype)
    thresh = random.randint(10, 40)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                new_image[y, x, c] = np.clip(image[y, x, c] + thresh, 0, 255)

    cv2.imwrite(returnFileName(counter), new_image)


def contrast(image):
    thresh = random.uniform(1.0, 1.25)
    new_image = cv2.convertScaleAbs(image, alpha=thresh, beta=0)

    cv2.imwrite(returnFileName(counter), new_image)


def zoom(img):
    thresh = random.random()

    while thresh < 0.95:
        thresh = random.random()

    h, w = img.shape[:2]
    h_taken = int(thresh * h)
    w_taken = int(thresh * w)
    h_start = random.randint(0, h - h_taken)
    w_start = random.randint(0, w - w_taken)
    new_image = img[h_start:h_start + h_taken, w_start:w_start + w_taken, :]

    cv2.imwrite(returnFileName(counter), new_image)


def rotation(img):
    angle = int(random.uniform(-6, 6))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w / 2), int(h / 2)), angle, 1)
    new_image = cv2.warpAffine(img, M, (w, h))

    cv2.imwrite(returnFileName(counter), new_image)


def gausianBlur(image):
    thresh = random.random()
    new_image = cv2.GaussianBlur(image, (5, 5), thresh)

    cv2.imwrite(returnFileName(counter), new_image)


def erosion_image(image):
    thresh = int(random.uniform(0, 2))
    kernel = np.ones((thresh, thresh), np.uint8)
    new_image = cv2.erode(image, kernel, iterations=1)

    cv2.imwrite(returnFileName(counter), new_image)


def sharpen(img):
    contrast = int(random.uniform(0, 15))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img[:, :, 2] = [[max(pixel - contrast, 0) if pixel < 190 else min(pixel + contrast, 255) for pixel in row] for row in img[:, :, 2]]
    new_image = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    cv2.imwrite(returnFileName(counter), new_image)


for IMG in os.listdir("./images/"):

    # Checking for image types
    if IMG.split(".")[1] not in ["png", "jpeg", "jpg"]:
        continue

    # Loading images
    image = cv2.imread("./images/" + IMG)

    # Resizing images to 500x300
    image = cv2.resize(image, (500, 300))

    # Applying each filter to an image
    # & updating their filenames after every save

    for _ in range(10):

        brighten(image)
        counter += 1

        print(counter)

        contrast(image)
        counter += 1

        print(counter)

        zoom(image)
        counter += 1

        print(counter)

        rotation(image)
        counter += 1

        print(counter)

        gausianBlur(image)
        counter += 1

        print(counter)

        erosion_image(image)
        counter += 1

        print(counter)

        sharpen(image)
        counter += 1

        print(counter)
