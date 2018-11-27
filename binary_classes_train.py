import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from tqdm import tqdm
sys.path.insert(0, 'Processes\\Back-end')
from models import make_models as mm
from models import BinaryClassNet

'''TRAINING THE BINARY CLASSIFICATION NET'''

Path = "Datasets\\Binary_Classification\\"

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = dsets.ImageFolder(Path, transform=transform)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=0)

device = torch.device('cpu')
Net = BinaryClassNet().to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(Net.parameters(), lr=0.001)

'''TRAINING THE MODEL'''

Epochs = 50
for epoch in range(Epochs):
    total = 0
    for i, (image, label) in enumerate(train_loader):
        try:
            output = Net(image.to(device))
            optimizer.zero_grad()
            loss = criterion(output, label.to(device))
            loss.backward()
            optimizer.step()
            total += loss.item()
            if i % 10 == 0 and i != 0:
                print('Epoch : {}/{}, Iteration = {}, Loss = {}, Avg. Loss = {}'.format(
                    epoch, Epochs, i, loss.item(), total / i))
        except Exception as e:
            print(e)

torch.save(Net.state_dict(), 'Processes\\Back-end\\Saved Model\\ClassificationNetCPU.pt')
