import os
import glob
import sys
from argparse import ArgumentParser
import numpy as np
import torch.nn.functional as F
import torch
from torchvision import transforms
from Models import *
# from Functions import TrainDataset
import torch.utils.data as Data
from data import datasets, trans
from natsort import natsorted
import csv
parser = ArgumentParser()
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--bs", type=int,
                    dest="bs", default=1, help="batch_size")
parser.add_argument("--iteration", type=int,
                    dest="iteration", default=320001,
                    help="number of total iterations")
parser.add_argument("--local_ori", type=float,
                    dest="local_ori", default=1000.0,
                    help="Local Orientation Consistency loss: suggested range 1 to 1000")
parser.add_argument("--magnitude", type=float,
                    dest="magnitude", default=0.001,
                    help="magnitude loss: suggested range 0.001 to 1.0")
parser.add_argument("--mask_labda", type=float,
                    dest="mask_labda", default=0.25,
                    help="mask_labda loss: suggested range 0.1 to 10")
parser.add_argument("--data_labda", type=float,
                    dest="data_labda", default=0.02,
                    help="data_labda loss: suggested range 0.1 to 10")
parser.add_argument("--smth_labda", type=float,
                    dest="smth_labda", default=0.02,
                    help="labda loss: suggested range 0.1 to 10")
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=403,
                    help="frequency of saving models")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=8,
                    help="number of start channels")
parser.add_argument("--datapath", type=str,
                    dest="datapath",default='../brain/OASIS/',
                    help="data path for training images")
parser.add_argument("--trainingset", type=int,
                    dest="trainingset", default=4,
                    help="1 Half : 200 Images, 2 The other Half 200 Images 3 All 400 Images")
parser.add_argument("--using_l2", type=int,
                    dest="using_l2",
                    default=1,
                    help="using l2 or not")
opt = parser.parse_args()

lr = opt.lr
bs = opt.bs
iteration = opt.iteration
start_channel = opt.start_channel
local_ori = opt.local_ori
magnitude = opt.magnitude
n_checkpoint = opt.checkpoint
smooth = opt.smth_labda
datapath = opt.datapath
mask_labda = opt.mask_labda
data_labda = opt.data_labda
trainingset = opt.trainingset
using_l2 = opt.using_l2

def dice(pred1, truth1):
    VOI_lbls = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 34, 36]
    dice_35=np.zeros(len(VOI_lbls))
    index = 0
    for k in VOI_lbls:
        #print(k)
        truth = truth1 == k
        pred = pred1 == k
        intersection = np.sum(pred * truth) * 2.0
        # print(intersection)
        dice_35[index]=intersection / (np.sum(pred) + np.sum(truth))
        index = index + 1
    return np.mean(dice_35)

def save_checkpoint(state, save_dir, save_filename, max_model_num=20):
    torch.save(state, save_dir + save_filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    # print(model_lists)
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

def Validate():
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    
    atlas_dir = datapath + '/atlas.pkl'
    train_dir = datapath + '/Train/'
    val_dir = datapath + '/Val/'
    train_composed = transforms.Compose([trans.RandomFlip(0),
                                         trans.NumpyType((np.float32, np.float32)),
                                         ])

    val_composed = transforms.Compose([trans.Seg_norm(), #rearrange segmentation label to 1 to 46
                                       trans.NumpyType((np.float32, np.int16))])
    train_set = datasets.IXIBrainDataset(glob.glob(train_dir + '*.pkl'), atlas_dir, transforms=train_composed)
    val_set = datasets.IXIBrainInferDataset(glob.glob(val_dir + '*.pkl'), atlas_dir, transforms=val_composed)
    train_loader = Data.DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = Data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)


    model = Cascade(2, 3, start_channel).to(device)
    if using_l2 == 1:
        loss_similarity = MSE().loss
    elif using_l2 == 0:
        loss_similarity = SAD().loss
    elif using_l2 == 2:
        loss_similarity = NCC()
    loss_smooth = smoothloss
#    loss_magnitude = magnitude_loss
#    loss_Jdet = neg_Jdet_loss

    transform = SpatialTransform().to(device)
    # init_dict = torch.load('/bask/projects/d/duanj-ai-imaging/Accreg/brain/Learn_FFT_40_48_56/LRL2ss_1_Chan_8_Smth_1000000.0_LR_0.0001/SYMNet_160000.pth')
    diff_transform = DiffeomorphicTransform(time_step=7).to(device)


    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True
    # for name, param in model.named_parameters():
        # param.requires_grad = True
        # with torch.no_grad():
            # if name in init_dict.keys():
                # print(name)
                # param.copy_(init_dict[name])

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = './L2ss_{}_Chan_{}_Smth_{}_LR_{}_All/'.format(using_l2,start_channel,smooth,lr)
    model_dir_val = './L2ss_{}_Chan_{}_Smth_{}_LR_{}_Val/'.format(using_l2,start_channel,smooth,lr)
    csv_name = 'L2ss_{}_Chan_{}_Smth_{}_LR_{}.csv'.format(using_l2,start_channel,smooth,lr)
    # assert os.path.exists(csv_name) ==0
    # assert os.path.isdir(model_dir) ==1
    # assert os.path.isdir(model_dir_val) ==0
    f = open(csv_name, 'w')
    with f:
        fnames = ['Index','Dice']
        writer = csv.DictWriter(f, fieldnames=fnames)
        writer.writeheader()

    if not os.path.isdir(model_dir_val):
        os.mkdir(model_dir_val)

    lossall = np.zeros((3, iteration))
    # train_set = TrainDataset(datapath,img_file='train_list.txt',trainingset = trainingset)
    # training_generator = Data.DataLoader(dataset=train_set, batch_size=1,
                                         # shuffle=True, num_workers=4)
    step = 1
    epoch = 0
    for epoch in range(1000):
        print("Validate " + model_dir + ' Epoch_{:09d}.pth'.format(epoch))
        try:
            current_model = torch.load(model_dir + 'Epoch_{:09d}.pth'.format(epoch))#['state_dict']
        except:
            print('Load model exception!')
            import time
            time.sleep(600)
            current_model = torch.load(model_dir + 'Epoch_{:09d}.pth'.format(epoch))#['state_dict']
        model.load_state_dict(current_model)
        with torch.no_grad():
            
            Dices_Validation = []
            Dices_Validation_Bilinear = []
            for data in val_loader:
                model.eval()
                xv = data[0]
                yv = data[1]
                xv_seg = data[2]
                yv_seg = data[3]
                v_xy = model(xv.float().to(device), yv.float().to(device))
                Dv_xy = diff_transform(v_xy)
                # Dv_xy = v_xy
                warped_xv_seg= transform(xv_seg.float().to(device), Dv_xy.permute(0, 2, 3, 4, 1), mod = 'nearest')
                
                # x_seg_oh = nn.functional.one_hot(xv_seg.long(), num_classes=46)
                # x_seg_oh = x_seg_oh.squeeze(1).permute(0, 4, 1, 2, 3).contiguous()
                # x_segs = transform(x_seg_oh.float().to(device), vf_xy.permute(0, 2, 3, 4, 1))
                # warped_xv_seg_bilinear = torch.argmax(x_segs, dim=1, keepdim=True)
                
                
                for bs_index in range(bs):
                    dice_bs=dice(warped_xv_seg[bs_index,...].data.cpu().numpy().copy(),yv_seg[bs_index,...].data.cpu().numpy().copy())
                    Dices_Validation.append(dice_bs)
                    # dice_bs=dice(warped_xv_seg_bilinear[bs_index,...].data.cpu().numpy().copy(),yv_seg[bs_index,...].data.cpu().numpy().copy())
                    # Dices_Validation_Bilinear.append(dice_bs)
            modelname = 'DiceVal_{:.5f}_Epoch_{:09d}.pth'.format(np.mean(Dices_Validation),epoch)
            f = open(csv_name, 'a')
            with f:
                writer = csv.writer(f)
                writer.writerow([epoch, np.mean(Dices_Validation)])
            save_checkpoint(model.state_dict(), model_dir_val, modelname)
Validate()
