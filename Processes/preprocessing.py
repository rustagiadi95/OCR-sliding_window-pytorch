'''
The data obtained from sliding window will be pre-processed here
'''

from skimage.transform import resize
import torchvision.transforms as transforms

# Here the file will get resized to 28X28 resized
def resize28X28(image_list):
    return [(resize(image[0], (28, 28), mode="constant"), image[1]) for image in image_list]

#Converting the images to tensor and normalizing the image
def transform_to_tensor(image_list):
    image_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    image_list = [(image_transform(image[0]), image[1])
                  for image in image_list]
    return image_list
