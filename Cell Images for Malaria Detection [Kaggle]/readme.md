Malaria Cell Images Dataset

FINAL Test Score: 92.11

url: https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria

The dataset contains 2 folders - Infected - Uninfected and a total of 27,558 images.
Note: Please download the above dataset and place "Parasitized" and "Uninfected" folder inside "cell_images" folder.

Methodology used: A simple CNN with 600k parameters (approx) in 5 Conv layers and FC layers, all connected to each other.

Implemented using python programming language, modules used are numpy, tensorflow, PIL.Image, cv2, scikit-learn, os, math

#########################################################################################
Input: (None, 64, 64, 3)
Output: (None, 1)
#########################################################################################
If needed (or experimentational purposes) apply StandardScaler to input images

No. of epochs = 35, learning_rate = 0.000025

FINAL Score (train): 98.22
FINAL Score (test): 92.11



* get data from backup folder or kaggle.com