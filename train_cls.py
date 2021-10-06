import numpy as np
import torch
import os
import model
import dataset
import utils
from  torchvision import transforms
from path import Path
import random
from torch.utils.data import Dataset, DataLoader
from args import parse_args




manualSeed = random.randint(1, 1000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

def PointNetLoss(outputs, labels, m3x3, m64x64, alpha=0.0001):
    criterion = torch.nn.NLLLoss()
    bs = outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs, 1, 1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs, 1, 1)
    if outputs.is_cuda:
        id3x3 = id3x3.cuda()
        id64x64 = id64x64.cuda()
    diff3x3 = id3x3 - torch.bmm(m3x3, m3x3.transpose(1, 2))
    diff64x64 = id64x64 - torch.bmm(m64x64, m64x64.transpose(1, 2))
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3)+torch.norm(diff64x64))/float(bs)


def train(args):
    path = Path(args.root_dir)
    folders = [dir for dir in sorted(os.listdir(path)) if os.path.isdir(path/dir)]
    classes = {folder: i for i, folder in enumerate(folders)}

    train_transforms = transforms.Compose([
        utils.Normalize(),
        utils.ToTensor()
    ])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    pointnet = model.PointNet().to(device)
    optimizer = torch.optim.Adam(pointnet.parameters(), lr=args.lr)

    train_dataset = dataset.pcddata(path, transform=train_transforms)
    valid_dataset = dataset.pcddata(path, valid=True, folder='test', transform=train_transforms)
    print('Train dataset size: ', len(train_dataset))
    print('Valid dataset size: ', len(valid_dataset))
    print('number of classes: ', len(train_dataset.classes))

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size)

    try:
        os.mkdir(args.saved_model)
    except OSError as error:
        print(error)

    print('Start traning!')
    for epoch in range(args.epochs):
        pointnet.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
            optimizer.zero_grad()
            outputs, m3x3, m64x64 = pointnet(inputs.transpose(1, 2))
            loss = PointNetLoss(outputs, labels, m3x3, m64x64)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:
                print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f' % (epoch + 1, i+1, len(train_loader), running_loss / 10))
                running_loss = 0.0

        pointnet.eval()
        correct = total = 0

        if valid_loader:
            with torch.no_grad():
                for data in valid_loader:
                    inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    outputs, __, __ = pointnet(inputs.transpose(1, 2))
                    __, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_acc = 100.* correct / total
            print('Validation accuracy: %d %%' % val_acc)

            checkpoint = Path(args.saved_model)/'save_'+str(epoch)+'.pth'
            torch.save(pointnet.state_dict(), checkpoint)
            print('Model saved to ', checkpoint)


if __name__ == '__main__':
    args = parse_args()
    train(args)





