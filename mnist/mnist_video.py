from __future__ import print_function
import argparse
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
from utils_for_DL import create_data_tuple, save_results_to_date_file
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import random
import cv2
from EOTF import EMD
from datetime import datetime


# 'runs', comment=datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
# setup the writer as global


class CustomImageDataset(Dataset):
    def __init__(self, dataset_dict, transform=None, target_transform=None):
        self.dataset_dict = dataset_dict
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset_dict)

    def __getitem__(self, idx):
        names = list(self.dataset_dict.keys())
        name_idx = names[idx]
        images = self.dataset_dict[name_idx][0]
        segmentation = self.dataset_dict[name_idx][1]
        if self.transform:
            images = self.transform(images)
        if self.target_transform:
            segmentation = self.target_transform(segmentation)
        return images, segmentation


class Net(nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__()
        # args that we can change:
        self.in_frame_dim = kwargs.setdefault('in_frame_dim', (4, 9, 220, 120))
        self.out_frame_dim = kwargs.setdefault('out_frame_dim', (4, 9, 220, 120))

        # for conv3d layer 1:
        self.stride1 = kwargs.setdefault('stride1', (1, 3, 3))
        self.padding1 = kwargs.setdefault('padding1', (0, 0, 0))
        self.dilation1 = kwargs.setdefault('dilation1', (1, 1, 1))
        self.Cout1 = kwargs.setdefault('Cout1', 9)
        self.conv1_kernel_size = kwargs.setdefault('conv1_kernel_size', (1, 3, 3))

        # for conv3d layer 2:
        self.stride2 = kwargs.setdefault('stride2', (1, 1, 1))
        self.padding2 = kwargs.setdefault('padding2', (0, 0, 0))
        self.dilation2 = kwargs.setdefault('dilation2', (1, 1, 1))
        self.Cout2 = kwargs.setdefault('Cout2', 32)
        self.conv2_kernel_size = kwargs.setdefault('conv1_kernel_size', (3, 3, 3))

        # for maxpool 3D layer:
        self.maxpool = kwargs.setdefault('maxpool', (1, 3, 3))
        self.maxpool_stride = kwargs.setdefault('maxpool_stride', self.maxpool)

        # for Linear layer 2:
        self.linear2_input = kwargs.setdefault('linear2_input', 5280)

        # -----------------------------------------------------------------------------
        # calculating important dims for the network initiation:
        # the format is (D, H, W) the letter is for the layer type (c for conv) and the index for the number:
        self.conv1_dims = torch.floor((torch.tensor(self.in_frame_dim)[1::] +
                                       2 * torch.tensor(self.padding1) - torch.tensor(self.dilation1) * (
                    torch.tensor(self.conv1_kernel_size) - 1) - 1) / torch.tensor(self.stride1) + 1)

        self.conv2_dims = torch.floor(
            ((torch.tensor(self.conv1_dims) + 2 * torch.tensor(self.padding2) - torch.tensor(self.dilation2) * (
                    torch.tensor(self.conv2_kernel_size) - 1) - 1) / torch.tensor(self.stride2)) + 1)

        self.max_pool1_dims = torch.floor(
            ((torch.tensor(self.conv2_dims) - (torch.tensor(self.maxpool) - 1) - 1) / torch.tensor(self.maxpool_stride)) + 1)

        self.linear1_input = int(self.Cout2 * torch.prod(self.max_pool1_dims))

        self.output_flatten_size = int(self.out_frame_dim[1] * self.out_frame_dim[2] * self.out_frame_dim[3])
        # -----------------------------------------------------------------------------
        # the network layers:
        self.conv1 = nn.Conv3d(self.in_frame_dim[0], self.Cout1, self.conv1_kernel_size, stride=self.stride1,
                               padding=self.padding1,
                               dilation=self.dilation1)
        self.conv2 = nn.Conv3d(self.Cout1, self.Cout2, self.conv2_kernel_size, stride=self.stride2,
                               padding=self.padding2,
                               dilation=self.dilation2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self.linear1_input, self.linear2_input)
        self.fc2 = nn.Linear(self.linear2_input, self.output_flatten_size)

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
        x = x.reshape((x.shape[0], self.out_frame_dim[1], self.out_frame_dim[2], self.out_frame_dim[3]))
        output = x  # torch.sigmoid(x) # torch.round(torch.sigmoid(x)) #
        return output


def logs_for_writer(model_tmp, epoch_num):
    # Layer = conv1
    writer.add_histogram("weights conv1 layer", model_tmp.conv1.weight.data.flatten(), epoch_num)
    writer.add_histogram("bias conv1 layer", model_tmp.conv1.bias.data.flatten(), epoch_num)
    if model_tmp.conv1.weight.grad is not None:  # At the first run it doesnt work
        writer.add_histogram("weights conv1 grad", model_tmp.conv1.weight.grad.flatten(), epoch_num)

    # Layer = conv2
    writer.add_histogram("weights conv2 layer", model_tmp.conv2.weight.data.flatten(), epoch_num)
    writer.add_histogram("bias conv2 layer", model_tmp.conv2.bias.data.flatten(), epoch_num)
    if model_tmp.conv2.weight.grad is not None:
        writer.add_histogram("weights conv2 grad", model_tmp.conv2.weight.grad.flatten(), epoch_num)

    # Layer = fc1
    writer.add_histogram("weights fc1 layer", model_tmp.fc1.weight.data.flatten(), epoch_num)
    writer.add_histogram("bias fc1 layer", model_tmp.fc1.bias.data.flatten(), epoch_num)
    if model_tmp.fc1.weight.grad is not None:
        writer.add_histogram("weights fc1 grad", model_tmp.fc1.weight.grad.flatten(), epoch_num)

    # Layer = fc2
    writer.add_histogram("weights fc2 layer", model_tmp.fc2.weight.data.flatten(), epoch_num)
    writer.add_histogram("bias fc2 layer", model_tmp.fc2.bias.data.flatten(), epoch_num)
    if model_tmp.fc2.weight.grad is not None:
        writer.add_histogram("weights fc2 grad", model_tmp.fc2.weight.grad.flatten(), epoch_num)


def batchify(data_dict: dict, batch_size: int) -> list:
    # to make the dict into list, and then couple them together, the return have to be
    # a list with the number of batches in every primary index
    elements = list(data_dict.values())
    random.shuffle(elements)

    list_of_raw_data = [data[0] for data in elements]
    list_of_results = [data[1] for data in elements]
    tensors_raw_data = torch.Tensor(np.array(list_of_raw_data)).type(torch.DoubleTensor)
    tensors_results = torch.Tensor(np.array(list_of_results)).type(torch.DoubleTensor)
    batched_list = list()


    idx = 0
    while idx < len(elements):
        tmp_list = list([tensors_raw_data[idx:idx + batch_size], tensors_results[idx:idx + batch_size]])
        batched_list.append(tmp_list)
        idx += batch_size

    return batched_list


def train(args, model, device, train_loader, optimizer, epoch, batch_size=1):
    model.train()
    print('\033[95m' + '-----------------------------------------------------------------------------\n' + '\033[0m')
    print(
        '\033[96m' + f'This current option is for 2 kernels of Fourier (x,y) '
                     f'and 2 in Glider.\nEach frame is ({input_dict["in_frame_dim"][2]}, {input_dict["in_frame_dim"][3]}).\n' + '\033[0m')
    print('\033[95m' + '-----------------------------------------------------------------------------\n' + '\033[0m')
    running_loss = 0.0

    train_list = batchify(train_loader, batch_size=batch_size)
    for batch_idx, (data, target) in enumerate(train_list):
        # ----------------------------------------------------------------------------------------------------------#
        # TODO: when we build the DATALOADER we need to delete the lines below.
        #       This current option is for 2 kernels of Fourier (x, y) and 2 in Glider
        # data, target = torch.tensor(np.array(data)), torch.tensor(np.array(target))
        # data, target = torch.unsqueeze(data, dim=0), torch.unsqueeze(target, dim=0)
        # data, target = data.type(torch.DoubleTensor), target.type(torch.DoubleTensor)

        # ----------------------------------------------------------------------------------------------------------#
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader),
                       100. * batch_idx / len(train_loader), loss.item()))
            writer.add_scalar('training loss', loss.item() / args.log_interval,
                              epoch * len(train_list) + batch_idx)

            if args.dry_run:
                break


def net_test(model, device, test_loader, transform):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for idx, (data, target) in enumerate(list(test_loader.values())):
            data, target = torch.tensor(np.array(data)), torch.tensor(np.array(target))
            data, target = torch.unsqueeze(data, dim=0), torch.unsqueeze(target, dim=0)
            data, target = data.type(torch.DoubleTensor), target.type(torch.DoubleTensor)
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = nn.L1Loss()
            loss = loss(output, target)
            test_loss += loss  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            # to put the output in the results dict:
            output_numpy = output.clone().cpu().numpy()
            # devide into frames:
            tmp_list = list()
            for i in range(output_numpy.shape[1]):
                tmp_list.append(output_numpy[:, i, :, :][0])
            results_dict[list(test_loader.keys())[idx]].append(tmp_list)
    test_loss /= len(test_loader)
    print(f'The test loss is: {test_loss}')


def main(model_in_parameters=dict()):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=2, metavar='N',
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

    global writer
    writer_comment = f' epochs = {args.epochs} ||  model_in_parameters ={model_in_parameters["in_frame_dim"]} ||  ' \
                     f' out_frame_dim = {model_in_parameters["out_frame_dim"]}'
    writer = SummaryWriter(comment=writer_comment)
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

    model = Net(**model_in_parameters).double().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    print('Before pre process')
    # Preprocess for train data
    for k in train_dict.keys():
        frames_orig = train_dict[k][0]
        frames_orig = [f / 255 for f in frames_orig]  # for the normalization

        frames_preprocessed = [
            frames_orig[:-1],
            EMD.forward_video(frames_orig, EMD.TEMPLATE_FOURIER, axis=0),
            EMD.forward_video(frames_orig, EMD.TEMPLATE_FOURIER, axis=1),
            EMD.forward_video(frames_orig, EMD.TEMPLATE_GLIDER, axis=0),
            EMD.forward_video(frames_orig, EMD.TEMPLATE_GLIDER, axis=1)
        ]
        train_dict[k] = (frames_preprocessed, train_dict[k][1])

    # Preprocess for test data
    for k in test_dict.keys():
        frames_orig = test_dict[k][0]
        frames_orig = [f / 255 for f in frames_orig]  # for the normalization

        frames_preprocessed = [
            frames_orig[:-1],
            EMD.forward_video(frames_orig, EMD.TEMPLATE_FOURIER, axis=0),
            EMD.forward_video(frames_orig, EMD.TEMPLATE_FOURIER, axis=1),
            EMD.forward_video(frames_orig, EMD.TEMPLATE_GLIDER, axis=0),
            EMD.forward_video(frames_orig, EMD.TEMPLATE_GLIDER, axis=1)
        ]
        test_dict[k] = (frames_preprocessed, test_dict[k][1])

    print('After pre process')
    data, target = list(train_dict.values())[0]
    data, target = torch.tensor(np.array(data)), torch.tensor(np.array(target))
    data, target = torch.unsqueeze(data, dim=0), torch.unsqueeze(target, dim=0)
    data, target = data.type(torch.DoubleTensor), target
    data, target = data.to(device), target.to(device)
    # data, target = data.repeat(4, 1, 1, 1), target # 4,9,40,40

    # tmp normalization:
    # data, target = data / 255, target / 255
    # data, target = torch.unsqueeze(data, dim=0), torch.unsqueeze(target, dim=0)

    writer.add_graph(model, data)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        logs_for_writer(model, epoch)
        train(args, model, device, train_dict, optimizer, epoch, batch_size=args.batch_size)  # dataset_dict
        net_test(model, device, test_dict, transform=transform)
        scheduler.step()
        elapsed = time.time() - epoch_start_time
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    import os

    save_data = 1
    # import wandb
    #
    # wandb.init(project="test-project", entity="yuvalandchen")

    # If you want to run it you need to extract the dataset from the zip
    # general_DS_folser = os.path.join('D:\Data_Sets', 'DAVIS-2017-trainval-480p', 'DAVIS')
    general_DS_folser = os.path.join('DAVIS-2017-trainval-480p', 'DAVIS')
    desired_dim = (120, 120)

    # The dataset format is:
    # {'name_of_video', raw jpegs, segmented data}
    number_of_videos = 90
    dataset_dict = create_data_tuple(general_DS_folser, number_of_videos=number_of_videos, desiered_dim=desired_dim,
                                     number_of_frames=9)

    # To create random train and test sets :
    indexes = np.arange(number_of_videos)
    np.random.shuffle(indexes)

    train_indexes = indexes[0:int(np.round(0.8 * number_of_videos))]
    test_indexes = indexes[int(np.round(0.8 * number_of_videos)) + 1::]

    names_list = list(dataset_dict.keys())
    train_dict = {}
    test_dict = {}
    results_dict = {}

    for idx in train_indexes:
        train_dict[names_list[idx]] = dataset_dict[names_list[idx]]

    for idx in test_indexes:
        test_dict[names_list[idx]] = dataset_dict[names_list[idx]]
        results_dict[names_list[idx]] = list()

    input_dict = {'in_frame_dim': (5, 8, desired_dim[0], desired_dim[1]),
                  'out_frame_dim': (1, 9, desired_dim[0], desired_dim[1]), 'Cout2': 8}

    # TODO: this is the start of data loader, still we need to built all of it's attributes
    loader = CustomImageDataset(dataset_dict)

    main(input_dict)
    writer.flush()
    writer.close()

    if save_data:
        save_results_to_date_file(results_dict)
    """
    Things that we need to talk about:
    1. what is the resolution that we put in the net
    2. how many channels -option1->  F:x,y ; G:x,y  -option2->  option1 + other_2*x,y
    3. should we start with a segmentaion of a single frame or to start directly with couple of frames. - Answer: video
    """
