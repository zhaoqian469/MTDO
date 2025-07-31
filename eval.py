from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from modules.MTDO_original import myNet as Net   #改这里！！
from functools import reduce
from data.data import get_eval_set

import numpy as np
import scipy.io as sio
import time
import cv2
import math

import pdb
import os

from skimage.metrics import structural_similarity

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Test settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--chop_forward', type=bool, default=False)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='D:\ZhaoQian\datasets\JILIN')
parser.add_argument('--file_list', type=str, default='Jilin_test_5.txt')   #改这里！！！
parser.add_argument('--other_dataset', type=bool, default=True, help="use other dataset than vimeo-90k")
parser.add_argument('--future_frame', type=bool, default=True, help="use future frame")
parser.add_argument('--nFrames', type=int, default=5)   #改这里！！
parser.add_argument('--model_type', type=str, default='MTDO')
parser.add_argument('--residual', type=bool, default=False)
parser.add_argument('--output', default='Results/JILIN_Y/MTDO', help='Location to save checkpoint models')  #改这里！！
parser.add_argument('--model', default='weights/MTDO_original/best_4x_MTDO_original_epoch_16.pth',
                    help='sr pretrained base model')  #改这里！！

opt = parser.parse_args()

gpus_list = range(opt.gpus)
print(opt)

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
test_set = get_eval_set(opt.data_dir, opt.nFrames, opt.upscale_factor, opt.file_list)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model ', opt.model_type)
if opt.model_type == 'MTDO':
    model = Net(nframes=opt.nFrames)
model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(opt.model, map_location=lambda storage, loc: storage).items()})

print('Pre-trained SR model is loaded.')
if cuda:
    model = model.cuda(gpus_list[0])

def eval():
    model.eval()   #切换到评估模式
    count = 0
    folder = 1   #测试集testset为1-5
    avg_psnr_predicted = 0.0
    avg_ssim_predicted = 0.0
    best_psnr_predicted = 0.0
    best_ssim_predicted = 0.0
    avg_psnr = 0.0
    avg_ssim = 0.0
    best_count_psnr = 0
    best_count_ssim = 0
    for batch in testing_data_loader:
        gt, input, neigbor, bicubic = batch[0], batch[1], batch[2], batch[3]

        with torch.no_grad():
            gt = Variable(gt).cuda(gpus_list[0])
            input = Variable(input).cuda(gpus_list[0])
            bicubic = Variable(bicubic).cuda(gpus_list[0])
            neigbor = [Variable(j).cuda(gpus_list[0]) for j in neigbor]



        t0 = time.time()
        if opt.chop_forward:
            with torch.no_grad():
                prediction = my_chop_forward(input, neigbor, model, opt.upscale_factor)
        else:
            with torch.no_grad():
                prediction = model(input, neigbor)
        
        if opt.residual:
            prediction = prediction + bicubic
            
        t1 = time.time()
        print("===> Processing: %s || Timer: %.4f sec." % (str(count), (t1 - t0)))
        img_name = '{0:08d}'.format(count)
        folder_name = '{0:03d}'.format(folder)
        save_img(prediction.cpu().data, img_name, folder_name, False)   #改这里！
        # save_img(target, str(count), False)

        prediction = prediction.cpu()
        prediction = prediction.data[0].numpy().astype(np.float32)
        prediction = prediction*255.

        target = gt.cpu().squeeze().numpy().astype(np.float32)
        target = target*255.

        prediction = np.transpose(prediction, (1, 2, 0))
        prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2YCrCb)
        target = np.transpose(target, (1, 2, 0))
        target = cv2.cvtColor(target, cv2.COLOR_BGR2YCrCb)
        # 提取 Y 通道
        prediction = prediction[:, :, 0]
        target = target[:, :, 0]

        #计算psnr
        psnr_predicted = PSNR(prediction,target, shave_border=opt.upscale_factor)
        avg_psnr_predicted += psnr_predicted
        avg_psnr += psnr_predicted

        #计算ssim
        ssim_predicted = structural_similarity(prediction,target,channel_axis=0, data_range=target.max() - target.min())
        avg_ssim_predicted += ssim_predicted
        avg_ssim += ssim_predicted

        #找到每个场景中psnr最好的图像
        if psnr_predicted > best_psnr_predicted:
            best_psnr_predicted = psnr_predicted
            best_count_psnr = count

        if ssim_predicted > best_ssim_predicted:
            best_ssim_predicted = ssim_predicted
            best_count_ssim = count

        if (count) == 99:
            count = -1
            folder = folder + 1

            #获取每个场景中重建效果psnr最好的图像路径
            save_dir = os.path.join(opt.output, os.path.splitext(opt.file_list)[0] + '_' + str(
                opt.upscale_factor) + 'x' + '/' + folder_name)
            save_fn = save_dir + '/' + str(best_count_psnr) + '.png'
            save_fn2 = save_dir + '/' + str(best_count_ssim) + '.png'

            txt_file = os.path.join(opt.output, "psnr-ssim_file_names.txt")
            os.makedirs(opt.output, exist_ok=True)

            with open(txt_file, "a") as f:
                f.write(save_fn + "\n"+"psnr:"+str(best_psnr_predicted)+"\n"+save_fn2 + "\n"+"ssim:"+str(best_ssim_predicted)+"\n"
                        +"avg_psnr:"+str(avg_psnr/100)+"\n"+"avg_ssim:"+str(avg_ssim/100)+"\n"+"\n")

            best_psnr_predicted = 0.0
            best_count_psnr = 0
            best_ssim_predicted = 0.0
            best_count_ssim = 0
            avg_psnr = 0.0
            avg_ssim = 0.0

        count+=1
    # #
    print("PSNR_predicted=", avg_psnr_predicted/len(testing_data_loader))   #所有图像的平均值
    print("SSIM_predicted=", avg_ssim_predicted/len(testing_data_loader))


def save_img(img, img_name, folder_name, pred_flag):
    save_img = img.squeeze().clamp(0, 1).numpy().transpose(1,2,0)

    # save img
    save_dir=os.path.join(opt.output, os.path.splitext(opt.file_list)[0]+'_'+str(opt.upscale_factor)+'x'+'/'+folder_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    if pred_flag:
        save_fn = save_dir +'/'+ img_name+'_'+opt.model_type+'F'+str(opt.nFrames)+'.png'
    else:
        save_fn = save_dir +'/'  + img_name+ '.png'
    cv2.imwrite(save_fn, cv2.cvtColor(save_img*255, cv2.COLOR_BGR2RGB),  [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print(save_fn)

def PSNR(pred, gt, shave_border=0):
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

def my_chop_forward(x, neigbor, model, scale, shave=8, min_size=2000, nGPUs=opt.gpus):
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    inputlist = [
        [x[:, :, 0:h_size, 0:w_size], [j[:, :, 0:h_size, 0:w_size] for j in neigbor]],
        [x[:, :, 0:h_size, (w - w_size):w], [j[:, :, 0:h_size, (w - w_size):w] for j in neigbor]],
        [x[:, :, (h - h_size):h, 0:w_size], [j[:, :, (h - h_size):h, 0:w_size] for j in neigbor]],
        [x[:, :, (h - h_size):h, (w - w_size):w], [j[:, :, (h - h_size):h, (w - w_size):w] for j in neigbor]]]

    if w_size * h_size < min_size:
        outputlist = []
        for i in range(0, 4, nGPUs):
            with torch.no_grad():
                input_batch = inputlist[i]#torch.cat(inputlist[i:(i + nGPUs)], dim=0)
                # output_batch = model(input_batch[0], input_batch[1], input_batch[2])
                output_batch = model(input_batch[0], input_batch[1])
            outputlist.extend(output_batch.chunk(nGPUs, dim=0))
    else:
        outputlist = [
            chop_forward(patch[0], patch[1], patch[2], model, scale, shave, min_size, nGPUs) \
            for patch in inputlist]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    with torch.no_grad():
        output = Variable(x.data.new(b, c, h, w))
    output[:, :, 0:h_half, 0:w_half] \
        = outputlist[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = outputlist[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = outputlist[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = outputlist[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output
    
def chop_forward(x, neigbor, flow, model, scale, shave=8, min_size=2000, nGPUs=opt.gpus):
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    inputlist = [
        [x[:, :, 0:h_size, 0:w_size], [j[:, :, 0:h_size, 0:w_size] for j in neigbor], [j[:, :, 0:h_size, 0:w_size] for j in flow]],
        [x[:, :, 0:h_size, (w - w_size):w], [j[:, :, 0:h_size, (w - w_size):w] for j in neigbor], [j[:, :, 0:h_size, (w - w_size):w] for j in flow]],
        [x[:, :, (h - h_size):h, 0:w_size], [j[:, :, (h - h_size):h, 0:w_size] for j in neigbor], [j[:, :, (h - h_size):h, 0:w_size] for j in flow]],
        [x[:, :, (h - h_size):h, (w - w_size):w], [j[:, :, (h - h_size):h, (w - w_size):w] for j in neigbor], [j[:, :, (h - h_size):h, (w - w_size):w] for j in flow]]]

    if w_size * h_size < min_size:
        outputlist = []
        for i in range(0, 4, nGPUs):
            with torch.no_grad():
                input_batch = inputlist[i]#torch.cat(inputlist[i:(i + nGPUs)], dim=0)
                output_batch = model(input_batch[0], input_batch[1], input_batch[2])
            outputlist.extend(output_batch.chunk(nGPUs, dim=0))
    else:
        outputlist = [
            chop_forward(patch[0], patch[1], patch[2], model, scale, shave, min_size, nGPUs) \
            for patch in inputlist]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    with torch.no_grad():
        output = Variable(x.data.new(b, c, h, w))
    output[:, :, 0:h_half, 0:w_half] \
        = outputlist[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = outputlist[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = outputlist[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = outputlist[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output

##Eval Start!!!!
eval()
