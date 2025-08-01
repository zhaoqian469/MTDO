from __future__ import print_function
import argparse
from math import log10

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from modules.MTDO import myNet as Net
from data.data import get_training_set, get_eval_set
import pdb
import socket
import time
import math
import numpy as np
import shutil

from tensorboardX import SummaryWriter

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=4, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch for continuing training')
parser.add_argument('--nEpochs', type=int, default=35, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=1, help='Snapshots')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.01')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
# Training set------------------------------------------------------------
parser.add_argument('--data_dir', type=str, default='D:\ZhaoQian\datasets\JILIN')
parser.add_argument('--file_list', type=str, default='Jilin_train_140.txt')
parser.add_argument('--other_dataset', type=bool, default=True, help="use other dataset than vimeo-90k")
# Test while training------------------------------------------------------------
parser.add_argument('--test_dir', type=str, default='D:\ZhaoQian\datasets\JILIN')
parser.add_argument('--test_file_list', type=str, default='jilin_val_40.txt')
#-------------------------------------------------------------------
parser.add_argument('--future_frame', type=bool, default=True, help="use future frame")
parser.add_argument('--nFrames', type=int, default=5)
parser.add_argument('--patch_size', type=int, default=64, help='0 to use original frame size')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--model_type', type=str, default='MTDO')
parser.add_argument('--residual', type=bool, default=False)
parser.add_argument('--pretrained_sr', default='')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--save_folder', default='weights/MTDO/', help='Location to save checkpoint models')
parser.add_argument('--prefix', default='', help='Location to save checkpoint models')

opt = parser.parse_args()
gpus_list = range(opt.gpus)

hostname = str(socket.gethostname())
cudnn.benchmark = True
print(opt)

writer = SummaryWriter('ablation/MTDO')

def PSNR(pred, gt, shave_border=0):
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %f M' % (num_params / 1e6))


def checkpoint(epoch):
    """
    Save the model weights of the current epoch.
    """
    if not os.path.exists(opt.save_folder):
        os.makedirs(opt.save_folder)

    model_out_path = os.path.join(
        opt.save_folder,
        f"{opt.upscale_factor}x_{hostname}{opt.model_type}{opt.prefix}_epoch_{epoch}.pth"
    )
    torch.save(model.state_dict(), model_out_path)
    print(f"Checkpoint saved to {model_out_path}")

def save_best_model(bestepoch):
    """
    Find the corresponding weight model file based on the best epoch, and copy it as the best model.
    """
    best_model_path = os.path.join(
        opt.save_folder,
        f"{opt.upscale_factor}x_{hostname}{opt.model_type}{opt.prefix}_epoch_{bestepoch}.pth"
    )

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Model file for epoch {bestepoch} not found: {best_model_path}")

    model_out_path = os.path.join(
        opt.save_folder,
        f"best_{opt.upscale_factor}x_{opt.model_type}_epoch_{bestepoch}.pth"
    )
    shutil.copy(best_model_path, model_out_path)
    print(f"BestModel saved to {model_out_path}")

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set(opt.data_dir, opt.nFrames, opt.upscale_factor, opt.data_augmentation, opt.file_list,
                             opt.patch_size)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

eval_set = get_eval_set(opt.test_dir, opt.nFrames, opt.upscale_factor, opt.test_file_list)
eval_data_loader = DataLoader(dataset=eval_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model ', opt.model_type)
if opt.model_type == 'MTDO':
    model = Net(nframes=opt.nFrames)

model = torch.nn.DataParallel(model, device_ids=gpus_list)
criterion = nn.L1Loss()

print('---------- Networks architecture -------------')
print_network(model)
print('----------------------------------------------')

if opt.pretrained:
    model_name = os.path.join(opt.save_folder + opt.pretrained_sr)
    if os.path.exists(model_name):
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        print('Pre-trained SR model is loaded.')

if cuda:
    model = model.cuda(gpus_list[0])
    criterion = criterion.cuda(gpus_list[0])

optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

best_epoch = 0
best_test_psnr = 0.0
for epoch in range(opt.start_epoch, opt.nEpochs + 1):
    epoch_loss = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        gt, input, neigbor, bicubic = batch[0], batch[1], batch[2], batch[3]

        if cuda:
            gt = Variable(gt).cuda(gpus_list[0])
            input = Variable(input).cuda(gpus_list[0])
            bicubic = Variable(bicubic).cuda(gpus_list[0])
            neigbor = [Variable(j).cuda(gpus_list[0]) for j in neigbor]

        optimizer.zero_grad()
        t0 = time.time()

        prediction = model(input, neigbor)

        if opt.residual:
            prediction = prediction + bicubic

        loss = criterion(prediction, gt)
        t1 = time.time()
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration,
                                                                                 len(training_data_loader), loss.item(),
                                                                                 (t1 - t0)))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
    writer.add_scalar('Avg. Loss', epoch_loss / len(training_data_loader), epoch)

    # test while training
    count = 1
    avg_psnr_predicted = 0.0
    avg_test_psnr = 0.0
    model.eval()
    for batch in eval_data_loader:
        gt, input, neigbor, bicubic = batch[0], batch[1], batch[2], batch[3]

        with torch.no_grad():
            gt = Variable(gt).cuda(gpus_list[0])
            input = Variable(input).cuda(gpus_list[0])
            bicubic = Variable(bicubic).cuda(gpus_list[0])
            neigbor = [Variable(j).cuda(gpus_list[0]) for j in neigbor]

        t0 = time.time()
        with torch.no_grad():
            prediction = model(input, neigbor)

        if opt.residual:
            prediction = prediction + bicubic

        t1 = time.time()
        print("===> Processing: %s || Timer: %.4f sec." % (str(count), (t1 - t0)))

        prediction = prediction.cpu()
        prediction = prediction.data[0].numpy().astype(np.float32)
        prediction = prediction * 255.

        gt = gt.cpu()
        gt = gt.squeeze().numpy().astype(np.float32)
        gt = gt * 255.

        psnr_predicted = PSNR(prediction, gt, shave_border=opt.upscale_factor)
        print(psnr_predicted)
        avg_psnr_predicted += psnr_predicted
        avg_test_psnr = avg_psnr_predicted / len(eval_data_loader)
        count += 1

    print("===> Epoch {} Complete: Avg. PSNR: {:.4f}".format(epoch, avg_psnr_predicted / len(eval_data_loader)))  #指的是一次迭代所有场景的平均值
    if avg_test_psnr > best_test_psnr:
        best_epoch = epoch
        best_test_psnr = avg_test_psnr
    if epoch == opt.nEpochs:
        print('Best_epoch:{:.4f},Best_psnr={:.6f}'.format(best_epoch, best_test_psnr))
        save_best_model(best_epoch)

    writer.add_scalar('Avg. PSNR', avg_psnr_predicted / len(eval_data_loader), epoch)


    # learning rate is decayed by a factor of 10 every half of total epochs
    if (epoch+1) % (10) == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 2.0
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

    if (epoch+1) % (opt.snapshots) == 0:
        checkpoint(epoch)
