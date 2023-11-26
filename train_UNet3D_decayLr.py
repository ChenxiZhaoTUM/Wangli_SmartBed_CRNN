import sys, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from model_UNet3D import UNet3D
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
saveMSELoss = True
# add Dropout2d layer?
dropout = 0

prefix = "UNet3D_01_"
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
netG = UNet3D(channelExponent=expo, dropout=dropout)
print(netG)  # print full net
model_parameters = filter(lambda p: p.requires_grad, netG.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Initialized UNet3D with {} trainable params ".format(params))
print()

if len(doLoad) > 0:
    # netG.load_state_dict(torch.load(doLoad))
    netG.load_state_dict(torch.load(doLoad, map_location=torch.device('cpu')))
    print("Loaded model " + doLoad)
netG.to(device)

criterionMSELoss = nn.MSELoss()
optimizer = optim.Adam(netG.parameters(), lr=lrG, weight_decay=0.0)
scheduler = StepLR(optimizer, step_size=decayLr, gamma=0.1)

##### training begins #####
for epoch in range(epochs):
    print("Starting epoch {} / {}".format((epoch + 1), epochs))

    # TRAIN
    netG.train()
    MSELoss_accum = 0.0
    for i, traindata in enumerate(trainLoader, 0):
        inputs_cpu, targets_cpu = traindata
        inputs, targets = inputs_cpu.to(device), targets_cpu.to(device)

        # test code
        # print(i)
        # print(inputs_cpu.size())  # torch.Size([50, 10, 12, 32, 64])
        # print(targets_cpu.size())  # torch.Size([50, 1, 32, 64])

        gen_out = netG(inputs)
        gen_out_cpu = gen_out.data.cpu().numpy()

        gen_out = gen_out.float()
        targets = targets.float()

        loss = criterionMSELoss(gen_out, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # LR decay
        if decayLr:
            scheduler.step()

        MSELossViz = loss.item()
        MSELoss_accum += MSELossViz

        if i == len(trainLoader) - 1:
            logline = "Epoch: {}, batch-idx: {}, MSELoss: {}\n".format(epoch, i, MSELossViz)
            print(logline)

        targets_denormalized = raw_dataset.denormalize(targets_cpu.cpu().numpy())
        outputs_denormalized = raw_dataset.denormalize(gen_out_cpu)

        random_indices = random.sample(range(len(trainLoader)), 10)
        if epoch % 500 == 0 and i in random_indices:
            for j in range(batch_size):
                utils.makeDirs(["TRAIN_UNet3D"])
                utils.imageOut("TRAIN_UNet3D/epoch{}_{}_{}".format(epoch, i, j), inputs_cpu[j],
                               targets_denormalized[j], outputs_denormalized[j])

            torch.save(netG.state_dict(), prefix + "model")

    # VALIDATION
    netG.eval()
    MSELossVal_accum = 0.0
    with torch.no_grad():
        for i, validata in enumerate(valiLoader, 0):
            inputs_cpu, targets_cpu = validata
            inputs, targets = inputs_cpu.to(device), targets_cpu.to(device)

            outputs = netG(inputs)
            outputs_cpu = outputs.data.cpu().numpy()

            outputs = outputs.float()
            targets = targets.float()

            loss = criterionMSELoss(outputs, targets)
            MSELossViz = loss.item()
            MSELossVal_accum += MSELossViz

            targets_denormalized = raw_dataset.denormalize(targets_cpu.cpu().numpy())
            outputs_denormalized = raw_dataset.denormalize(outputs_cpu)

            random_indices = random.sample(range(len(valiLoader)), 10)
            if epoch % 500 == 0 and i in random_indices:
                for j in range(batch_size):
                    utils.makeDirs(["VALIDATION_UNet3D"])
                    utils.imageOut("VALIDATION_UNet3D/epoch{}_{}_{}".format(epoch, i, j), inputs_cpu[j],
                                   targets_denormalized[j], outputs_denormalized[j])

    MSELoss_accum /= len(trainLoader)
    MSELossVal_accum /= len(valiLoader)
    if saveMSELoss:
        if epoch == 0:
            utils.resetLog(prefix + "MSELoss.txt")
            utils.resetLog(prefix + "MSELossVal.txt")
        utils.log(prefix + "MSELoss.txt", "{} ".format(MSELoss_accum), False)
        utils.log(prefix + "MSELossVal.txt", "{} ".format(MSELossVal_accum), False)
