# importing necessary files
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from google.colab.patches import cv2_imshow
from sklearn.metrics.pairwise import cosine_similarity

# Locations to the files
image1_location = os.listdir("./images/")[0]
image2_location = os.listdir("./images/")[1]

# Hard coded values of embeddings
sizes = [[[5, 0], [175, 90]],
         [[225, 0], [395, 90]],
         [[440, 0], [485, 90]],
         [[55, 95], [225, 200]],
         [[270, 95], [440, 200]],
         [[5, 205], [175, 305]],
         [[225, 205], [395, 305]],
         [[440, 205], [485, 305]]
		]

# Main fucntion
def main_func(img1_location, img2_location):
  # Value initialization
  sum = 0

  # Loading in the images
  image1 = cv2.imread(img1_location, 0)
  image2 = cv2.imread(img2_location, 0)

  # Resizing in the images
  image1 = cv2.resize(image1, (2500, 1550))
  image2 = cv2.resize(image2, (2500, 1550))

  print("Image 1")
  cv2_imshow(image1)

  print("Image 2")
  cv2_imshow(image2)

  # Looping over the sizes array and adding the Cosine similarity between both images at same patches
  for size in sizes:
    sum += 1 - cosine_similarity(image1[size[0][1]:size[1][1], size[0][0]:size[1][0]].reshape(1,-1), image2[size[0][1]:size[1][1], size[0][0]:size[1][0]].reshape(1,-1))

  print("Sum:", sum)

  average = sum / 8

  print("Average:", average)

  return average

main_func(image1_location, image2_location)