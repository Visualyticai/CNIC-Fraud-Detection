# Modules
import os
import cv2
import random
import numpy as np

# file number counter
counter = 1


def returnFileName(temp):
    return "./Generated/Generated-" + str(temp) + ".png"


def jpegCompression(image):
  thresh = randint(0, 75)

  cv2.imwrite(returnFileName(counter), image, [int(cv2.IMWRITE_JPEG_QUALITY), thresh])


def gausianBlur(image):
    thresh = random.random()
    new_image = cv2.GaussianBlur(image, (5, 5), thresh)

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
