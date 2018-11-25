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


def get_name_num(image):
    name_li = sliding_window(image)
    return name_li
