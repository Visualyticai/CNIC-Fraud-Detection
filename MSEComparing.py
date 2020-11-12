# importing necessary files
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Locations to the files
image1_location = "./images/image1.jpg"
image2_location = "./images/image2.jpeg"

print(image1_location, image2_location)

# Hard coded values of embeddings
sizes = [[[5, 0], [175, 90]],
         [[225, 0], [395, 90]],
         [[440, 0], [485, 90]],
         [[55, 95], [225, 200]],
         [[270, 95], [440, 200]],
         [[5, 205], [175, 305]], s
         [[225, 205], [395, 305]],
         [[440, 205], [485, 305]]
         ]

# Helper Function
def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


# Main Function
def ComparingImagesUsingMSE(img1_location, img2_location):
    # Value initialization
    sum = 0

    # Loading in the images
    image1 = cv2.imread(img1_location, 0)
    image2 = cv2.imread(img2_location, 0)

    # Resizing in the images
    image1 = cv2.resize(image1, (500, 300))
    image2 = cv2.resize(image2, (500, 300))

    # cv2_imshow(image1)

    # cv2_imshow(image2)

    # Looping over the sizes array and adding the Cosine similarity between both images at same patches
    for size in sizes:
        fig = plt.figure()

        ax = fig.add_subplot(1, 2, 1)
        plt.imshow(image1[size[0][1]:size[1][1], size[0][0]:size[1][0]], cmap=plt.cm.gray)
        ax = fig.add_subplot(1, 2, 2)
        plt.imshow(image2[size[0][1]:size[1][1], size[0][0]:size[1][0]], cmap=plt.cm.gray)

        plt.show()

        temp = mse(image1[size[0][1]:size[1][1], size[0][0]:size[1][0]].reshape(1, -1), image2[size[0][1]:size[1][1], size[0][0]:size[1][0]].reshape(1, -1))
        sum += temp

        print(size, "--->", temp)
        print("\n\n")

    print("Sum:", sum)

    average = sum / 8

    # if (sum > 0.279):
    #   print("Embedding detected!")
    #   return

    print("Average:", average)

    return average

ComparingImagesUsingMSE(image1_location, image2_location)