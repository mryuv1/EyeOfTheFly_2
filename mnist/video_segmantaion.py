from __future__ import print_function
import time
from video_seg_parts import Up, Down, DoubleConv_3d, OutConv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from video_seg_utils import *
import os
import random
import cv2
from EOTF import EMD
from datetime import datetime

global yuvals_computer
yuvals_computer = 0

if not yuvals_computer:
    from torch.utils.tensorboard import SummaryWriter


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


class CreateDatasetArgs:
    def __init__(self, **kwargs):
        self.image_size = kwargs.setdefault('desired_dim', (128,128))
        self.number_of_videos = kwargs.setdefault('number_of_videos', np.inf)
        self.preprocess_type = kwargs.setdefault('preprocess_type', None)
        self.train_part = kwargs.setdefault('train_part', 0.8)
        self.video_length = kwargs.setdefault('video_length', 9)
        self.type = kwargs.setdefault('type', None)


class RunArgs:
    def __init__(self, **kwargs):
        self.batch_size = kwargs.setdefault('batch_size', 2)
        self.dry_run = kwargs.setdefault('dry_run', False)
        self.scheduler_gamma = kwargs.setdefault('gamma', 0.7)
        self.scheduler_step_size = kwargs.setdefault('step_size', 5)
        self.epochs = kwargs.setdefault('epochs', 14)
        self.log_interval = kwargs.setdefault('log_interval', 10)
        self.lr = kwargs.setdefault('lr', 40)
        self.no_cuda = kwargs.setdefault('no_cuda', False)
        self.seed = kwargs.setdefault('seed', False)

    def to_dict_form(self):
        dict_to_return = dict()
        dict_to_return['batch_size'] = self.batch_size
        dict_to_return['dry_run'] = self.dry_run
        dict_to_return['epochs'] = self.epochs
        dict_to_return['log_interval'] = self.log_interval
        dict_to_return['lr'] = self.lr
        dict_to_return['no_cuda'] = self.no_cuda
        dict_to_return['seed'] = self.seed

        return dict_to_return


class Net(nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__()
        # args that we can change:
        self.in_frame_dim = kwargs.setdefault('in_frame_dim', (9, 4, 220, 120))
        self.out_frame_dim = kwargs.setdefault('out_frame_dim', (9, 220, 120))
        self.out_channels_1 = kwargs.setdefault('out_channels_1', 32)

        # the network layers:
        """
        self.inc = DoubleConv_3d(in_channels=self.in_frame_dim[0], out_channels=self.out_channels_1)
        self.down1 = Down(in_channels=self.out_channels_1, out_channels=2 * self.out_channels_1)
        self.down2 = Down(in_channels=2 * self.out_channels_1, out_channels=4 * self.out_channels_1)
        self.down3 = Down(in_channels=4 * self.out_channels_1, out_channels=4 * self.out_channels_1)
        self.up1 = Up(in_channels=8 * self.out_channels_1, out_channels=2 * self.out_channels_1)
        self.up2 = Up(in_channels=4 * self.out_channels_1, out_channels=self.out_channels_1)
        self.up3 = Up(in_channels=2 * self.out_channels_1, out_channels=self.out_channels_1)
        self.outc = OutConv(self.out_channels_1, 1)
        """
        self.inc = DoubleConv_3d(in_channels=self.in_frame_dim[0], out_channels=self.out_channels_1)
        self.down1 = Down(in_channels=self.out_channels_1, out_channels=2 * self.out_channels_1)
        self.down2 = Down(in_channels=2 * self.out_channels_1, out_channels=2 * self.out_channels_1)
        self.up1 = Up(in_channels=4 * self.out_channels_1, out_channels=self.out_channels_1)
        self.up2 = Up(in_channels=2 * self.out_channels_1, out_channels=self.out_channels_1)
        self.outc = OutConv(self.out_channels_1, 1)
        # self.init_weights()

    # def init_weights(self):
    #     torch.nn.init.xavier_uniform_(self.conv1.weight)
    #     torch.nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, x):
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        output = self.outc(x)  # torch.sigmoid(x) # torch.round(torch.sigmoid(x)) #
        """
        x1 = self.inc(x) # channels = out_channenls_1
        x2 = self.down1(x1) # channels = 2 * out_channels_1
        x3 = self.down2(x2) # channels = 2 * out_channels_1
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        output = self.outc(x)  # torch.sigmoid(x) # torch.round(torch.sigmoid(x)) #
        output = output.squeeze(1)
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
    criterion = nn.BCEWithLogitsLoss()
    train_list = batchify(train_loader, batch_size)
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
        loss = criterion(output, target)
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [Batch {batch_idx + 1}/{len(train_list)}], Loss: {loss.item():.2f}')
            if args.dry_run:
                break
    running_loss /= len(train_list)
    return running_loss


def net_test(model, device, epoch, test_loader, args):
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
            test_loss += loss.item()  # sum up batch loss

            if epoch>13:
                # to put the output in the results dict:
                output_numpy = output.clone().cpu().numpy()
                # devide into frames:
                tmp_list = list()
                for i in range(output_numpy.shape[1]):
                    tmp_list.append(output_numpy[:, i, :, :][0])
                results_dict[list(test_loader.keys())[idx]].append(tmp_list)
    test_loss /= len(test_loader)
    return test_loss


def main(model_in_parameters=None, args=RunArgs()):
    if model_in_parameters is None:
        model_in_parameters = dict()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    model = Net(**model_in_parameters).double().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    data, target = list(train_dict.values())[0]
    data, target = torch.tensor(np.array(data)), torch.tensor(np.array(target))
    data, target = torch.unsqueeze(data, dim=0), torch.unsqueeze(target, dim=0)
    data, target = data.type(torch.DoubleTensor), target
    data, target = data.to(device), target.to(device)

    if not yuvals_computer:
        writer.add_graph(model, data)
    scheduler = StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_loss = train(args, model, device, train_dict, optimizer, epoch, batch_size=args.batch_size)  # dataset_dict
        epoch_end_time = time.time()
        scheduler.step()
        test_loss = net_test(model, device, epoch, test_dict, args=args)

        test_loss_dict[epoch] = test_loss
        train_loss_dict[epoch] = train_loss
        print(f'--- Test loss is {test_loss}. ---')
        print(f'--- Epoch time is {epoch_end_time-epoch_start_time} seconds. ---')
        print(f'--- Test time is {time.time()-epoch_end_time} seconds. ---')
        train_loss_dict[epoch] = train_loss
        if not yuvals_computer:
            writer.add_scalar('training loss', train_loss, epoch)
            writer.add_scalar('test loss', test_loss, epoch)
        print()
    # if args.save_model:
    #     torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':

    save_data = 1

    # If you want to run it you need to extract the dataset from the zip
    # general_DS_folder = os.path.join('D:\Data_Sets', 'DAVIS-2017-trainval-480p', 'DAVIS')
    general_DS_folder = os.path.join('DAVIS-2017-trainval-480p', 'DAVIS')

    create_dataset_args = CreateDatasetArgs()
    create_dataset_args.image_size = (128, 128)
    create_dataset_args.number_of_videos = 20
    create_dataset_args.type = 'float32'
    create_dataset_args.video_length = 1
    create_dataset_args.train_part = 0.8

    # The dataset format is:
    # {(raw jpegs, segmented data)}
    train_dict, test_dict, results_dict = create_dataset(general_DS_folder, create_dataset_args,
                                                         override_saved_data=True, save_new_data=False)

    input_dict = {'in_frame_dim': (8, create_dataset_args.video_length,
                                   create_dataset_args.image_size[0], create_dataset_args.image_size[1]),
                  'out_frame_dim': (1, create_dataset_args.video_length,
                                    create_dataset_args.image_size[0], create_dataset_args.image_size[1])}
    # input_dict format: (channels, length, height, width)

    run_args = RunArgs()
    run_args.epochs = 25
    run_args.scheduler_gamma = 0.3
    run_args.scheduler_step_size = 5
    run_args.lr = 0.1
    run_args.batch_size = 10

    print('\033[93m'
          + '-----------------------------------------------------------------------------\n'
          + '\033[0m')
    print('\033[94m'
          + f'The parameters for the run are: '
            f'gamma = ({run_args.scheduler_gamma}), lr = ({run_args.lr}), batch_size = ({run_args.batch_size}).\n'
          + '\033[0m')
    print('\033[93m'
          + '-----------------------------------------------------------------------------\n')

    test_loss_dict = dict()
    train_loss_dict = dict()

    # Run Training
    train_dict_orig = train_dict.copy()
    test_dict_orig = test_dict.copy()
    results_dict_orig = results_dict.copy()
    for preprocess_type in ['emd', 'duplicate', 'random_emd']:
        """
        if preprocess_type == 'duplicate':
            preprocess_func = duplicate_preprocess
        elif preprocess_type == 'emd':
            preprocess_func = emd_preprocess
        elif preprocess_type == 'random_emd':
            preprocess_func = random_emd_preprocess
        """
        print('starting preprocess: ' + preprocess_type)
        train_dict = preprocess(train_dict_orig.copy(), preprocess_type, True)
        test_dict = preprocess(test_dict_orig.copy(), preprocess_type, True)
        results_dict = results_dict_orig.copy()
        print('finished preprocess.')
        if not yuvals_computer:
            global writer
            writer_comment = f' epochs = {run_args.epochs} ||  model_in_parameters ={input_dict["in_frame_dim"]} ||  ' \
                             f' out_frame_dim = {input_dict["out_frame_dim"]} || preprocess_type = {preprocess_type}'
            writer = SummaryWriter(comment=writer_comment)
        main(input_dict, run_args)
        writer.flush()
        writer.close()

        if save_data:
            final_dict = {'run_params': run_args.to_dict_form(),
                          'test_dict': test_dict,
                          'results': results_dict,
                          'test_loss_dict': test_loss_dict,
                          'train_loss_dict': train_loss_dict}
            file_name = f'gamma = ({run_args.scheduler_gamma}), lr = ({run_args.lr}), batch_size = ({run_args.batch_size}), preprocess_type = ({preprocess_type})'
            save_results_to_date_file(final_dict, file_name=file_name)
