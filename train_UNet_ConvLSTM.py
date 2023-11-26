import sys, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from model_UNet_ConvLSTM import UNet_ConvLSTM
from SmartBedDataset_ReadOriginalTensor import SmartBedDataset_Base, SmartBedDataset_Train
import utils

use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

##### basic settings #####
# number of training iterations
iterations = 1000000
# batch size
batch_size = 50
# time step
time_step = 10
# learning rate
lrG = 0.0006
# decay learning rate?
decayLr = True
# channel exponent to control network size
expo = 3
# data set config
prop = None  # by default, use all from "./dataset/for_train"
# save txt files with per epoch loss?
saveL1 = True
# add Dropout2d layer?
dropout = 0

prefix = "UNet_ConvLSTM_01_"
if len(sys.argv) > 1:
    prefix = sys.argv[1]
    print("Output prefix: {}".format(prefix))

doLoad = ""  # optional, path to pre-trained model

print("LR: {}".format(lrG))
print("LR decay: {}".format(decayLr))
print("Iterations: {}".format(iterations))

seed = random.randint(0, 2 ** 32 - 1)
print("Random seed: {}".format(seed))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

##### create pytorch data object with dataset #####
raw_dataset = SmartBedDataset_Base(time_step=time_step)
train_dataset = SmartBedDataset_Train(raw_dataset, model="train")
trainLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
print("Training batches: {}".format(len(trainLoader)))

vali_dataset = SmartBedDataset_Train(raw_dataset, model="validation")
valiLoader = DataLoader(vali_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
print("Validation batches: {}".format(len(valiLoader)))
print()

##### setup training #####
epochs = iterations
netG = UNet_ConvLSTM(channelExponent=expo, dropout=dropout)
print(netG)  # print full net
model_parameters = filter(lambda p: p.requires_grad, netG.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Initialized UNet_ConvLSTM with {} trainable params ".format(params))
print()

if len(doLoad) > 0:
    # netG.load_state_dict(torch.load(doLoad))
    netG.load_state_dict(torch.load(doLoad, map_location=torch.device('cpu')))
    print("Loaded model " + doLoad)
netG.to(device)

criterionL1 = nn.L1Loss()
optimizerG = optim.Adam(netG.parameters(), lr=lrG, weight_decay=0.0)

##### training begins #####
for epoch in range(epochs):
    print("Starting epoch {} / {}".format((epoch + 1), epochs))

    # TRAIN
    netG.train()
    L1_accum = 0.0
    for i, traindata in enumerate(trainLoader, 0):
        inputs_cpu, targets_cpu = traindata
        inputs, targets = inputs_cpu.to(device), targets_cpu.to(device)

        # test code
        # print(i)
        # print(inputs_cpu.size())  # torch.Size([50, 10, 12, 32, 64])
        # print(targets_cpu.size())  # torch.Size([50, 1, 32, 64])

        # compute LR decay
        if decayLr:
            currLr = utils.computeLR(epoch, epochs, lrG * 0.1, lrG)
            if currLr < lrG:
                for g in optimizerG.param_groups:
                    g['lr'] = currLr

        gen_out = netG(inputs)
        gen_out_cpu = gen_out.data.cpu().numpy()

        lossL1 = criterionL1(gen_out, targets)
        optimizerG.zero_grad()
        lossL1.backward()
        optimizerG.step()

        lossL1viz = lossL1.item()
        L1_accum += lossL1viz

        if i == len(trainLoader) - 1:
            logline = "Epoch: {}, batch-idx: {}, L1: {}\n".format(epoch, i, lossL1viz)
            print(logline)

        targets_denormalized = raw_dataset.denormalize(targets_cpu.cpu().numpy())
        outputs_denormalized = raw_dataset.denormalize(gen_out_cpu)

        if lossL1viz < 0.01:
            for j in range(batch_size):
                utils.makeDirs(["TRAIN_UNet_ConvLSTM_0.01"])
                utils.imageOut("TRAIN_UNet_ConvLSTM_0.01/epoch{}_{}_{}".format(epoch, i, j), inputs[j],
                               targets_denormalized[j], outputs_denormalized[j])

        if lossL1viz < 0.01:
            torch.save(netG.state_dict(), prefix + "model")

    # VALIDATION
    netG.eval()
    L1val_accum = 0.0
    with torch.no_grad():
        for i, validata in enumerate(valiLoader, 0):
            inputs_cpu, targets_cpu = validata
            inputs, targets = inputs_cpu.to(device), targets_cpu.to(device)

            outputs = netG(inputs)
            outputs_cpu = outputs.data.cpu().numpy()

            lossL1 = criterionL1(outputs, targets)
            L1val_accum += lossL1.item()

            targets_denormalized = raw_dataset.denormalize(targets_cpu.cpu().numpy())
            outputs_denormalized = raw_dataset.denormalize(outputs_cpu)

            if lossL1viz < 0.01:
                for j in range(batch_size):
                    utils.makeDirs(["VALIDATION_UNet_ConvLSTM_0.01"])
                    utils.imageOut("VALIDATION_UNet_ConvLSTM_0.01/epoch{}_{}_{}".format(epoch, i, j), inputs[j],
                                   targets_denormalized[j], outputs_denormalized[j])

    L1_accum /= len(trainLoader)
    L1val_accum /= len(valiLoader)
    if saveL1:
        if epoch == 0:
            utils.resetLog(prefix + "L1.txt")
            utils.resetLog(prefix + "L1val.txt")
        utils.log(prefix + "L1.txt", "{} ".format(L1_accum), False)
        utils.log(prefix + "L1val.txt", "{} ".format(L1val_accum), False)
