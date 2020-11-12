# importing necessary files
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity

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

# Main Function
def CompareImagesUsingSSIM(image1_location, image2_location):
  # Initialization
  total = 0
  
  # Loading in the images
  image1 = cv2.imread(img1_location, 0)
  image2 = cv2.imread(img2_location, 0)

  # Resizing in the images
  image1 = cv2.resize(image1, (500, 300))
  image2 = cv2.resize(image2, (500, 300))

  # Convert images to grayscale
  image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
  image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

  for size in sizes:
    fig = plt.figure()

    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(image1[size[0][1]:size[1][1], size[0][0]:size[1][0]], cmap = plt.cm.gray)
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(image2[size[0][1]:size[1][1], size[0][0]:size[1][0]], cmap = plt.cm.gray)
    
    plt.show()
    
    (score, diff) = structural_similarity(image1[size[0][1]:size[1][1], size[0][0]:size[1][0]], image2[size[0][1]:size[1][1], size[0][0]:size[1][0]], full=True)
    total += score
    
    print(size, "--->", score)
    print("\n\n")


  print("Image similarity", total)

  return total

CompareImagesUsingSSIM(image1_location, image2_location)