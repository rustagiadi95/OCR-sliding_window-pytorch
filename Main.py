import sys
sys.path.insert(0, 'Channel_Name_Num')
sys.path.insert(0, '\\Back-end')
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
    3) Feed the slides to ocr.py which gives the detected text.
'''

binClass, classNet = mm()


def recog_name_number(image, CRN):

    csv = pd.read_csv('name and nums.csv')

    # Retrieving the co-ordinates of channel name and number
    df = csv[csv['CRN'] == CRN]
    num_pos = ast.literal_eval(df['Num Position'].values[0])
    name_pos = ast.literal_eval(df['Name Position'].values[0])

    # Localizing the channel name and number
    name_li, num_li = get_name_num(image, num_pos, name_pos)

    # Preprocessing, normlizing and converting the slide images to tensor
    name_li = transform_to_tensor(resize28X28(name_li))
    num_li = transform_to_tensor(resize28X28(num_li))

    #evaluating the channel name and number
    channel_num = ocr.evaluate(num_li, binClass, classNet, 0)
    channel_name = ocr.evaluate(name_li, binClass, classNet, 1)

    print('channel_name : ' + channel_name)
    print('channel_num : ' + channel_num)


CRN = 1059

image = 'D:\\Projects\\ArtifIQ\\channel_detection\\Data_New\\OCR1\\CRN\\CHR00' + \
    str(CRN) + '\\80.jpg'
image = cv2.imread(image)
# cv2.imshow('image', image)
# cv2.waitKey(0)
'''ENTRY POINT OF THE MODULE'''
# print(image.shape)
recog_name_number(image, CRN)
