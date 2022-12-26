from __future__ import print_function
import argparse
import sys
import time
from utils_modified_from_Unet import Up, Down, DoubleConv_3d, OutConv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
from utils_for_DL import create_data_tuple, save_results_to_date_file

global yuvals_computer
yuvals_computer = 1

from torch.utils.data import Dataset

if not yuvals_computer:
    from torch.utils.tensorboard import SummaryWriter
import random
import cv2
from EOTF import EMD
from datetime import datetime


# 'runs', comment=datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
# setup the writer as global


# class CustomImageDataset(Dataset):
#     def __init__(self, dataset_dict, transform=None, target_transform=None):
#         self.dataset_dict = dataset_dict
#         self.transform = transform
#         self.target_transform = target_transform
#
#     def __len__(self):
#         return len(self.dataset_dict)
#
#     def __getitem__(self, idx):
#         names = list(self.dataset_dict.keys())
#         name_idx = names[idx]
#         images = self.dataset_dict[name_idx][0]
#         segmentation = self.dataset_dict[name_idx][1]
#         if self.transform:
#             images = self.transform(images)
#         if self.target_transform:
#             segmentation = self.target_transform(segmentation)
#         return images, segmentation

def make_train_and_test_dicts(datasetPath, number_of_videos, desired_dim):
    dataset_dict = create_data_tuple(datasetPath, number_of_videos=number_of_videos, desiered_dim=desired_dim,
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

    return train_dict, test_dict, results_dict


class Args():
    def __init__(self, **kwargs):
        self.batch_size = kwargs.setdefault('batch_size', 2)
        self.dry_run = kwargs.setdefault('dry_run', False)
        self.gamma = kwargs.setdefault('gamma', 0.7)
        self.epochs = kwargs.setdefault('epochs', 14)
        self.log_interval = kwargs.setdefault('log_interval', 10)
        self.lr = kwargs.setdefault('epochs', 10)
        self.no_cuda = kwargs.setdefault('epochs', False)
        self.seed = kwargs.setdefault('seed', False)


class Net(nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__()
        # args that we can change:
        self.in_frame_dim = kwargs.setdefault('in_frame_dim', (9, 4, 220, 120))
        self.out_frame_dim = kwargs.setdefault('out_frame_dim', (9, 220, 120))
        self.out_channels_1 = kwargs.setdefault('out_channels_1', 32)


        # the network layers:
        self.inc = DoubleConv_3d(in_channels=self.in_frame_dim[0], out_channels=self.out_channels_1)
        self.down1 = Down(in_channels=self.out_channels_1, out_channels=2 * self.out_channels_1)
        self.down2 = Down(in_channels=2 * self.out_channels_1, out_channels=4 * self.out_channels_1)
        self.down3 = Down(in_channels=4 * self.out_channels_1, out_channels=4 * self.out_channels_1)
        self.up1 = Up(in_channels=8 * self.out_channels_1, out_channels=2 * self.out_channels_1)
        self.up2 = Up(in_channels=4 * self.out_channels_1, out_channels=self.out_channels_1)
        self.up3 = Up(in_channels=2 * self.out_channels_1, out_channels=self.out_channels_1)
        self.outc = OutConv(self.out_channels_1, 1)
        # self.init_weights()

    # def init_weights(self):
    #     torch.nn.init.xavier_uniform_(self.conv1.weight)
    #     torch.nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        output = self.outc(x)  # torch.sigmoid(x) # torch.round(torch.sigmoid(x)) #
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
    train_list = batchify(train_loader, batch_size=1)

    for batch_idx, (data, target) in enumerate(train_list):
        # ----------------------------------------------------------------------------------------------------------#
        # TODO: when we build the DATALOADER we need to delete the lines below.
        #       This current option is for 2 kernels of Fourier (x, y) and 2 in Glider
        # data, target = torch.tensor(np.array(data)), torch.tensor(np.array(target))
        # data, target = torch.unsqueeze(data, dim=0), torch.unsqueeze(target, dim=0)
        # data, target = data.type(torch.DoubleTensor), target.type(torch.DoubleTensor)
        # data, target = torch.squeeze(data, dim=0), torch.squeeze(target, dim=0)
        # ----------------------------------------------------------------------------------------------------------#
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = torch.squeeze(output, dim=0)
        criterion = nn.BCEWithLogitsLoss()
        #TODO:to fix the line below
        target = target[:,1:]
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader),
                       100. * batch_idx / len(train_loader), loss.item()))
            if not yuvals_computer:
                writer.add_scalar('training loss', loss.item() / args.log_interval,
                                  epoch * len(train_list) + batch_idx)

            if args.dry_run:
                break


def net_test(model, device, test_loader):
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
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(output, target)
            # loss = nn.L1Loss()
            # loss = loss(output, target)
            test_loss += loss  # sum up batch loss
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            # to put the output in the results dict:
            output_numpy = output.clone().cpu().numpy()
            # devide into frames:
            tmp_list = list()
            for i in range(output_numpy.shape[1]):
                tmp_list.append(output_numpy[:, i, :, :][0])
            results_dict[list(test_loader.keys())[idx]].append(tmp_list)
    test_loss /= len(test_loader)
    print(f'The test loss is: {test_loss}')


def main(model_in_parameters=dict(), run_args_dict=dict()):
    args = Args(**run_args_dict)
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if not yuvals_computer:
        global writer
        writer_comment = f' epochs = {args.epochs} ||  model_in_parameters ={model_in_parameters["in_frame_dim"]} ||  ' \
                         f' out_frame_dim = {model_in_parameters["out_frame_dim"]}'
        writer = SummaryWriter(comment=writer_comment)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

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

    if not yuvals_computer:
        writer.add_graph(model, data)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        # logs_for_writer(model, epoch)
        train(args, model, device, train_dict, optimizer, epoch, batch_size=args.batch_size)  # dataset_dict
        # net_test(model, device, test_dict)
        scheduler.step()
        elapsed = time.time() - epoch_start_time
    # if args.save_model:
    #     torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    import os

    save_data = 1
    # import wandb
    #
    # wandb.init(project="test-project", entity="yuvalandchen")

    # If you want to run it you need to extract the dataset from the zip
    general_DS_folder = os.path.join('D:\Data_Sets', 'DAVIS-2017-trainval-480p', 'DAVIS')
    # general_DS_folder = os.path.join('DAVIS-2017-trainval-480p', 'DAVIS')
    desired_dim = (60, 60)

    # The dataset format is:
    # {'name_of_video', raw jpegs, segmented data}
    number_of_videos = 4
    train_dict, test_dict, results_dict = make_train_and_test_dicts(general_DS_folder, number_of_videos, desired_dim)

    input_dict = {'in_frame_dim': (5, 8, desired_dim[0], desired_dim[1]),
                  'out_frame_dim': (1, 9, desired_dim[0], desired_dim[1]), 'Cout2': 8}
    run_args_dict = dict()

    main(input_dict, run_args_dict)
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
