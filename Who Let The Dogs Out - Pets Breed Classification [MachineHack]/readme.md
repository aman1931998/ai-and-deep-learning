
Who Let The Dogs Out - Pets Breed Classification Hackathon

FINAL Score: Not Evaluated

url: https://www.machinehack.com/course/who-let-the-dogs-out-pets-breed-classification-hackathon/

Hackathon Info: Did you know that a survey said that 94% of pet owners say their animal pal makes them smile more than once a day? There are no prizes for guessing which are the most popular pets on the planet: they are cats and dogs. But how do you choose which one to adopt? How do you know if you’re a cat person or a dog person? And finally, how do you choose which breed to adopt (because let’s face it, they all look cute)? To make matters more difficult, there are around 340 breeds recognized by the Fédération Cynologique Internationale (FCI), the governing body of dog breeds, which is also called World Canine Organisation. On the other hand, The International Cat Association (TICA) recognizes 58 standardized breeds of cats.

Methodology used: We had 5 breeds of each cat and dog, so we converted into a 10-class classification model. We divided 6206 images into two datasets of 5500 and 706 images for training and validation of model. 

Implemented using python programming language, modules used are numpy, pandas, cv2, tensorflow, scikit-learn and math.

#########################################################################################
No. of parameters: 6.4 million parameters (approx)
Approach -> CNN Model (resembling similarities with AlexNet), 7 Convolutional Layers.
Input = (None, 224, 224, 3)
Output = (None, 10)
#########################################################################################

FINAL Score: Not Evaluated.

Tensorboard Visualizing Graph Included: tensorboard --logdir "LOCATION_TO_FOLDER\tmp\tflogs"

