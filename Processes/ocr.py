import sys
import torch
sys.path.insert(0, 'Back-end')
import cv2

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
           'small_a', 'small_b', 'small_c', 'small_d', 'small_e', 'small_f', 'small_g', 'small_h', 'small_i', 'small_j', 'small_k', 'small_l', 'small_m', 'small_n', 'small_o', 'small_p',
           'small_q', 'small_r', 'small_s', 'small_t', 'small_u', 'small_v', 'small_w', 'small_x', 'small_y', 'small_z']


def evaluate(image_li, binclass, classNet, category):
    '''This loop will do the character segmentation and will filter out the slides not having the slides'''
    chars = []
    for image in image_li:
        '''Uncomment to see the slides in action'''
        # cv2.imshow('image', image[0].detach().numpy()[0])
        # cv2.waitKey(0)
        img = torch.tensor(image[0].unsqueeze(0), dtype=torch.double)
        output = binclass(img)
        if output[0][0] < output[0][1]:
            chars.append(image)

    '''This loop will predict the characters from the filtered slides'''
    string = ''
    previous_char = ''
    previous_i = -1
    for image in chars:
        img = torch.tensor(image[0].unsqueeze(0), dtype=torch.double)
        index = (classNet(img)[0] == max(classNet(img)[0])).nonzero().item()
        current_i = image[1]
        char = classes[index]
        if index > 35:
            char = char.split('_')[1]
        if previous_char == char and current_i == previous_i + 1:
            continue
        previous_char = char
        previous_i = current_i
        string = string + char
    return string
