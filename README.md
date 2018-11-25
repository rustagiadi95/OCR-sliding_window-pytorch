# OCR-sliding_window-pytorch

AN OCR USING THE SLIDING WINDOW APPROACH

File architecture and purpose :

  1) Main.py - Driver file, recog_image function takes the image path to predict text from the images.
     Uses :-
     
     2) Processes/create_slides.py - Creates the slides of the image using the get_name_num
     
     3) Processes/preprocessing.py - Resizes the images to 28x28x3, normalizes the image on mean, std-dev and transforms the image to tensor.
     
     4) Processes/ocr.py - Holds the logic to evaluate the image to string.
     
     5) Processes/Back-end/models.py - Holds the code of the models BinClass(image segmentation) and ClassificationNet(character recognition)
     
  Processes/Back-end/Saved_Model contains the .pt saved weights of both of the above models
  
UPCOMING CHANGES :- 
  1) Script for training the models(train.py)
  2) training, input, testing image samples
  3) Creating the custom dataset
  4) Making the repo scalable for fuzzy logics, searching etc.
  5) Demo Script
