# Modules
import os
import cv2
import random
import numpy as np

# File counter
counter = 0

class dataAugmentation:
    def __init__(self, image):
        self.image = image

    def returnFileName(self):
        global counter
        counter += 1
        return "./Generated/Generated-" + str(counter) + ".png"

    def jpegCompression(self, image):
        thresh = random.randint(0, 75)

        cv2.imwrite(self.returnFileName(), image, [int(cv2.IMWRITE_JPEG_QUALITY), thresh])

    def gausianBlur(self, image):
        thresh = random.random()
        new_image = cv2.GaussianBlur(image, (5, 5), thresh)

        cv2.imwrite(self.returnFileName(), new_image)

    def gaussianNoise(self, image):
        row, col, ch = image.shape
        mean = random.randint(1, 10)
        var = random.randint(200, 500)
        sigma = var ** 0.5

        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)

        new_image = image + gauss

        cv2.imwrite(self.returnFileName(), new_image)

    def speckleNoise(self, image):
        row, col, ch = image.shape

        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)

        thresh = random.randint(7, 20)

        new_image = image + image * gauss * 1 / thresh

        cv2.imwrite(self.returnFileName(), new_image)

    def SnPNoise(self, image):
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = random.uniform(0.002, 0.01)
        new_image = np.copy(image)

        num_salt = np.ceil(amount * image.size * s_vs_p)

        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]

        new_image[tuple(coords)] = 1

        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))

        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]

        new_image[tuple(coords)] = 0

        cv2.imwrite(self.returnFileName(), new_image)

    def poisonNoise(self, image):
        vals = random.uniform(0.2, 0.9)
        vals = 2 ** np.ceil(np.log2(vals))

        new_image = np.random.poisson(image * vals) / float(vals)

        cv2.imwrite(self.returnFileName(), new_image)

    def brighten(self, image):
        new_image = np.zeros(image.shape, image.dtype)
        thresh = random.randint(10, 40)
        
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                for c in range(image.shape[2]):
                    new_image[y, x, c] = np.clip(image[y, x, c] + thresh, 0, 255)

        cv2.imwrite(self.returnFileName(), new_image)

    def contrast(self, image):
        thresh = random.uniform(1.0, 1.25)
        new_image = cv2.convertScaleAbs(image, alpha=thresh, beta=0)
        
        cv2.imwrite(self.returnFileName(), new_image)

    def zoom(self, img):
        thresh = random.random()
        
        while thresh < 0.95:
            thresh = random.random()

        h, w = img.shape[:2]
        h_taken = int(thresh * h)
        w_taken = int(thresh * w)
        h_start = random.randint(0, h - h_taken)
        w_start = random.randint(0, w - w_taken)
        new_image = img[h_start:h_start + h_taken, w_start:w_start + w_taken, :]

        cv2.imwrite(self.returnFileName(), new_image)

    def rotation(self, img):
        angle = int(random.uniform(-6, 6))
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((int(w / 2), int(h / 2)), angle, 1)
        new_image = cv2.warpAffine(img, M, (w, h))
        
        cv2.imwrite(self.returnFileName(), new_image)

    def erosion_image(self, image):
        thresh = int(random.uniform(0, 2))
        kernel = np.ones((thresh, thresh), np.uint8)
        new_image = cv2.erode(image, kernel, iterations=1)
        
        cv2.imwrite(self.returnFileName(), new_image)

    def sharpen(self, img):
        contrast = int(random.uniform(0, 15))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img[:, :, 2] = [[max(pixel - contrast, 0) if pixel < 190 else min(pixel + contrast, 255) for pixel in row] for row in img[:, :, 2]]
        new_image = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        
        cv2.imwrite(self.returnFileName(), new_image)


if __name__ == "__main__":
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
        
        DA = dataAugmentation(image)

        for _ in range(10):

            DA.jpegCompression(image)

            DA.gausianBlur(image)

            DA.gaussianNoise(image)

            DA.speckleNoise(image)

            DA.SnPNoise(image)

            DA.poisonNoise(image)

            DA.gausianBlur(image)
            
            print(counter)

            DA.brighten(image)
            
            DA.contrast(image)

            DA.zoom(image)

            DA.rotation(image)

            DA.gausianBlur(image)

            DA.erosion_image(image)

            DA.sharpen(image)

            print(counter)

