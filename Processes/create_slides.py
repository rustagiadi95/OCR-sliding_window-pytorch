import sys
sys.path.insert(0, 'Text_Recognition')
import matplotlib.pyplot as plt


def sliding_window(image):
    li = []
    height, width, channel = image.shape

    #TAKING THE WINDOW'S WIDTH AS 0.7 TIMES THE HEIGHT
    window_width = int(height*0.7)
    
    i = 0
    for items in range(0, width-window_width//3, window_width//3):
        img = image[:, items:items+window_width, :]
        li.append((img, i))
        i += 1
    return li


def get_name_num(image, num_pos, name_pos):
    name_li = sliding_window(image[name_y:name_yf, name_x:name_xf, :])
    num_li = sliding_window(image[num_y:num_yf, num_x:num_xf, :])
    return (name_li, num_li)
