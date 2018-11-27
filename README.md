# OCR-sliding_window-pytorch

AN OCR USING THE SLIDING WINDOW APPROACH

Repository architecture :

  1) Main.py - Driver file, recog_image function takes the image path to predict text from the images.
    Uses :-

     
    2) Processes/create_slides.py - Creates the slides of the image using the get_name_num.
     
    3) Processes/preprocessing.py - Resizes the images to 28x28x3, normalizes the image on mean, std-dev and transforms the image to tensor.
     
    4) Processes/ocr.py - Holds the logic to evaluate the image to string.
     
    5) Processes/Back-end/models.py - Holds the code of the models BinClass(image segmentation) and ClassificationNet(character recognition).

  6) character_classes_train.py - Trains the character classification model based on data stored in Datasets\Character_Classification folder. The file saves the trained model in Processes\Back-end\Saved_Model folder.

  7) binary_classes_train.py - Trains the binary classification(character segmentation) model on the data stored in 
  Datasets\Binary_Classification. The file saves the trained model in Processes\Back-end\Saved_Model folder.
     
  Processes/Back-end/Saved_Model contains the .pt saved weights of both of the above models

  Datasets folder contains dataset for two models :- 

  1) Binary Classification(character segmentation) model - Contains two folders namely Positive_Dataset and Negative_Dataset to
  classify that which of the slides has the character and which on does not.

  2) Character Classification model - Contains 62 folders (0-9(10) + a-z(26) + A-Z(26)) present as the classes to trains the ClassificationNet neural net.
  
UPCOMING CHANGES :- 
  1) training, input, testing image samples
  2) Making the repo scalable for fuzzy logics, searching etc.
  3) Demo Script
