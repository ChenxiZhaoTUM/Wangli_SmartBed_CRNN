import sys, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from model_CRNN import CRNN
from SmartBedDataset_ReadOriginalTensor_MeanAndStd import SmartBedDataset_Base, SmartBedDataset_Train
import utils_MeanAndStd as utils

use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

##### basic settings #####
# number of training iterations
iterations = 10000
# batch size
batch_size = 50
# time step
time_step = 10
# learning rate
lrG = 0.001
# decay learning rate?
decayLr = True
# channel exponent to control network size
expo = 4
# data set config
prop = None  # by default, use all from "./dataset/for_train"
# save txt files with per epoch loss?
saveMSELoss = True
# add Dropout2d layer?
dropout = 0.01

prefix = "CRNN_expo4_mean_01_"
if len(sys.argv) > 1:
    prefix = sys.argv[1]
    print("Output prefix: {}".format(prefix))

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
netG = CRNN(channelExponent=expo, dropout=dropout)
print(netG)  # print full net
model_parameters = filter(lambda p: p.requires_grad, netG.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Initialized CRNN with {} trainable params ".format(params))
print()

doLoad = ""  # optional, path to pre-trained model
if len(doLoad) > 0:
    # netG.load_state_dict(torch.load(doLoad))
    netG.load_state_dict(torch.load(doLoad, map_location=torch.device('cpu')))
    print("Loaded model " + doLoad)
netG.to(device)

criterionMSELoss = nn.MSELoss()
optimizer = optim.Adam(netG.parameters(), lr=lrG, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

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

        # # compute LR decay
        # if decayLr:
        #     currLr = utils.computeLR(epoch, epochs, lrG * 0.1, lrG)
        #     if currLr < lrG:
        #         for g in optimizer.param_groups:
        #             g['lr'] = currLr

        gen_out = netG(inputs)
        gen_out_cpu = gen_out.data.cpu().numpy()

        gen_out = gen_out.float()
        targets = targets.float()

        loss = criterionMSELoss(gen_out, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        MSELossViz = loss.item()
        MSELoss_accum += MSELossViz

        if i == len(trainLoader) - 1:
            logline = "Epoch: {}, batch-idx: {}, MSELoss: {}, Learning rate: {}\n".format(epoch, i, MSELossViz, lrG)
            print(logline)

        targets_denormalized = raw_dataset.denormalize(targets_cpu.cpu().numpy())
        outputs_denormalized = raw_dataset.denormalize(gen_out_cpu)

        random_indices = random.sample(range(len(trainLoader)), 10)
        if epoch % 500 == 0 and i in random_indices:
            for j in range(batch_size):
                utils.makeDirs(["TRAIN_CRNN_expo4_mean_01"])
                utils.imageOut("TRAIN_CRNN_expo4_mean_01/epoch{}_{}_{}".format(epoch, i, j), inputs_cpu[j],
                               targets_denormalized[j], outputs_denormalized[j])

            torch.save(netG.state_dict(), prefix + str(epoch) + "model")

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
                    utils.makeDirs(["VALIDATION_CRNN_expo4_mean_01"])
                    utils.imageOut("VALIDATION_CRNN_expo4_mean_01/epoch{}_{}_{}".format(epoch, i, j), inputs_cpu[j],
                                   targets_denormalized[j], outputs_denormalized[j])

    MSELoss_accum /= len(trainLoader)
    MSELossVal_accum /= len(valiLoader)
    if saveMSELoss:
        if epoch == 0:
            utils.resetLog(prefix + "MSELoss.txt")
            utils.resetLog(prefix + "MSELossVal.txt")
        utils.log(prefix + "MSELoss.txt", "{} ".format(MSELoss_accum), False)
        utils.log(prefix + "MSELossVal.txt", "{} ".format(MSELossVal_accum), False)

torch.save(netG.state_dict(), prefix + str(epochs) + "model")
