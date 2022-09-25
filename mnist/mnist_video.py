from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
from utils_for_DL import create_data_tuple


class Net(nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__()
        # args that we can change:
        self.frame_dim = kwargs.setdefault('frame_dim', (4, 9, 220, 120))
        self.kernel_size = kwargs.setdefault('kernel_size', 3)

        # for conv3d layer 1:
        self.stride1 = kwargs.setdefault('stride1', 1)
        self.padding1 = kwargs.setdefault('padding1', 0)
        self.dilation1 = kwargs.setdefault('dilation1', 1)
        self.Cout1 = kwargs.setdefault('Cout1', 16)

        # for conv3d layer 2:
        self.stride2 = kwargs.setdefault('stride2', 1)
        self.padding2 = kwargs.setdefault('padding2', 0)
        self.dilation2 = kwargs.setdefault('dilation2', 1)
        self.Cout2 = kwargs.setdefault('Cout2', 32)

        # for maxpool 3D layer:
        self.maxpool = kwargs.setdefault('maxpool', 2)
        self.maxpool_stride = kwargs.setdefault('maxpool_stride', 1)

        # for Linear layer 2:
        self.linear2_input = kwargs.setdefault('linear2_input', 5280)

        # -----------------------------------------------------------------------------
        # calculating important dims for the network initiation:
        # the format is (D, H, W) the letter is for the layer type (c for conv) and the index for the number:
        self.d_c1 = torch.floor(
            torch.tensor(((self.frame_dim[1] + 2 * self.padding1 - self.dilation1 * (
                        self.kernel_size - 1) - 1) / self.stride1) + 1))

        self.h_c1 = torch.floor(
            torch.tensor(((self.frame_dim[2] + 2 * self.padding1 - self.dilation1 * (
                        self.kernel_size - 1) - 1) / self.stride1) + 1))

        self.w_c1 = torch.floor(
            torch.tensor(((self.frame_dim[3] + 2 * self.padding1 - self.dilation1 * (
                        self.kernel_size - 1) - 1) / self.stride1) + 1))

        self.d_c2 = torch.floor(
            ((self.d_c1 + 2 * self.padding1 - self.dilation1 * (self.kernel_size - 1) - 1) / self.stride1) + 1)

        self.h_c2 = torch.floor(
            ((self.h_c1 + 2 * self.padding1 - self.dilation1 * (self.kernel_size - 1) - 1) / self.stride1) + 1)

        self.w_c2 = torch.floor(
            ((self.w_c1 + 2 * self.padding1 - self.dilation1 * (self.kernel_size - 1) - 1) / self.stride1) + 1)


        self.d_mp1 = torch.floor(
            ((self.d_c2 - (self.maxpool - 1) - 1) / self.maxpool_stride) + 1)

        self.h_mp1 = torch.floor(
            ((self.h_c2 - (self.maxpool - 1) - 1) / self.maxpool_stride) + 1)

        self.w_mp1 = torch.floor(
            ((self.w_c2 - (self.maxpool - 1) - 1) / self.maxpool_stride) + 1)

        self.linear1_input = int(self.Cout2 * self.d_mp1 * self.h_mp1 * self.w_mp1)

        self.output_flatten_size = int(self.frame_dim[1] * self.frame_dim[2] * self.frame_dim[3])
        # -----------------------------------------------------------------------------
        # the network layers:
        self.conv1 = nn.Conv3d(self.frame_dim[0], self.Cout1, self.kernel_size, stride=self.stride1, padding=self.padding1,
                               dilation=self.dilation1)
        self.conv2 = nn.Conv3d(self.Cout1, self.Cout2, self.kernel_size, stride=self.stride2, padding=self.padding2,
                               dilation=self.dilation2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self.linear1_input, self.linear2_input)
        self.fc2 = nn.Linear( self.linear2_input, self.output_flatten_size)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool3d(x, self.maxpool, stride=self.maxpool_stride)
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
    print('\033[95m' + '-----------------------------------------------------------------------------\n' + '\033[0m')
    print(
        '\033[96m' + f'This current option is for 2 kernels of Fourier (x,y) '
                     f'and 2 in Glider.\nEach frame is ({check["frame_dim"][2]}, {check["frame_dim"][3]}).\n' + '\033[0m')
    print('\033[95m' + '-----------------------------------------------------------------------------\n' + '\033[0m')
    for batch_idx, (data, target) in enumerate(list(train_loader.values())):
        # ----------------------------------------------------------------------------------------------------------#
        # TODO: when we build the DATALOADER we need to delete the lines below.
        #       This current option is for 2 kernels of Fourier (x, y) and 2 in Glider
        data, target = torch.tensor(np.array(data)), torch.tensor(np.array(target))
        data, target = torch.unsqueeze(data, dim=0), torch.unsqueeze(target, dim=0)
        data, target = data.repeat(4, 1, 1, 1), target

        # tmp normalization:
        data, target = data/255, target/255
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
            data, target = torch.tensor(np.array(data)), torch.tensor(np.array(target))
            data, target = torch.unsqueeze(data, dim=0), torch.unsqueeze(target, dim=0)
            data, target = data.repeat(4, 1, 1, 1), target

            # tmp normalization:
            data, target = data / 255, target / 255
            data, target = torch.unsqueeze(data, dim=0), torch.unsqueeze(target, dim=0)

            output = model(data)
            loss = nn.MSELoss()
            loss = loss(output, torch.flatten(target, 1))
            test_loss += loss  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

    test_loss /= len(test_loader)
    print(f'The test loss is: {test_loss}')


def main(model_in_parameters=dict()):
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

    model = Net(**model_in_parameters).to(device)
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

    desiered_dim = (40, 40)

    # The dataset format is:
    # {'name_of_video', raw jpegs, segmented data}
    dataset_dict = create_data_tuple(general_DS_folser, number_of_videos=40, desiered_dim=desiered_dim,
                                     number_of_frames=9)
    check = {'frame_dim': (4, 9, desiered_dim[0], desiered_dim[1]), 'Cout2': 16}

    main(check)

    """
    Things that we need to talk about:
    1. what is the resolution that we put in the net
    2. how many channels -option1->  F:x,y ; G:x,y  -option2->  option1 + other_2*x,y
    3. should we start with a segmentaion of a single frame or to start directly with couple of frames. - Answer: video
    """
