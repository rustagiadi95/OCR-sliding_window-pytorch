import sys
sys.path.insert(0, 'Processes')
sys.path.insert(0, 'Processes\\Back-end')
from create_slides import get_name_num
from preprocessing import resize28X28, transform_to_tensor
from models import make_models as mm
import ocr
import cv2

'''
This driver file will recieve the image path
Steps of flow :- 
    1) Create Slides of the image
    2) Does the preprocessing (resizing each slide to 28x28x3 and transform to tensor)
    3) Feed the preprocessed slides to ocr.py which gives the detected text.
'''

binClass, classNet = mm()

def recog_image(image_path):

    image = cv2.imread(image_path)

    slides = get_name_num(image)

    # Preprocessing, normlizing and converting the slides to tensor
    transformed_slides = transform_to_tensor(resize28X28(slides))

    #evaluating the image to text
    string = ocr.evaluate(transformed_slides, binClass, classNet, 1)

    print('Predicted string is : ' + string)
