from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from utils_for_DL import create_data_tuple


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 3)  # format: (in_channels, out_channels, kermel_size, stride)
        self.conv2 = nn.Conv2d(16, 32, 3, 3)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2304, 5280)
        self.fc2 = nn.Linear(5280, 26400)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = x
        return output


def train(args, model, device, train_loader, optimizer, epoch, transform):
    model.train()
    for batch_idx, (data, target) in enumerate(list(train_loader.values())):
        # ----------------------------------------------------------------------------------------------------------#
        # TODO: when we build the DATALOADER we need to delete the lines below
        data, target = data[0], target[0]
        data, target = transform(data), transform(target)
        data, target = torch.unsqueeze(data, dim=0), torch.unsqueeze(target, dim=0)
        # ----------------------------------------------------------------------------------------------------------#
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.MSELoss()
        loss = loss(output, torch.flatten(target, 1))
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader, transform):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for (data, target) in list(test_loader.values()):
            data, target = data[0], target[0]
            data, target = transform(data), transform(target)
            data, target = torch.unsqueeze(data, dim=0), torch.unsqueeze(target, dim=0)
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = nn.MSELoss()
            loss = loss(output, torch.flatten(target, 1))
            test_loss += loss  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

    test_loss /= len(test_loader)
    print(f'The test loss is: {test_loss}')



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, dataset_dict, optimizer, epoch, transform=transform)  # dataset_dict

        test(model, device, dataset_dict, transform=transform)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    import os
    # If you want to run it you need to extract the dataset from the zip
    general_DS_folser = os.path.join('D:\Data_Sets', 'DAVIS-2017-trainval-480p', 'DAVIS')

    # The dataset format is:
    # {'name_of_video', raw jpegs, segmented data}
    dataset_dict = create_data_tuple(general_DS_folser, number_of_videos=40, desiered_dim=(220, 120))
    main()

    """
    Things that we need to talk about:
    1. what is the resolution that we put in the net.
    2. how many channels.
    3. should we start with a segmentaion of a single frame or to start directly with couple of frames.
    """