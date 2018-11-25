'''
The file holds the architecture of the models involved.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dsets

'''
The code of negative / positive sample character segmentation net
'''

class BinaryClassNet(nn.Module):
    def __init__(self):
        super(BinaryClassNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.conv3 = nn.Conv2d(16, 36, 5)
        self.fc1 = nn.Linear(36 * 2 * 2, 60)
        self.fc2 = nn.Linear(60, 30)
        self.fc3 = nn.Linear(30, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(F.relu(self.conv3(x)), 2, 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


'''
The architecture of classification of characters neural network
'''
class ClassificationNet(nn.Module):

    def __init__(self):
        super(ClassificationNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 62)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def make_models():

    binclass = BinaryClassNet().double()
    binclass.load_state_dict(torch.load(
        "D:\Projects\ArtifIQ\channel_detection\OCR\Channel_Name_Num\Back-end\Saved_Model\ClassificationNetv3.pt"))

    classNet = ClassificationNet().double()
    classNet.load_state_dict(torch.load(
        "D:\Projects\ArtifIQ\channel_detection\OCR\Channel_Name_Num\Back-end\Saved_Model\MNISTNetCPU.pt"))

    return (binclass, classNet)
