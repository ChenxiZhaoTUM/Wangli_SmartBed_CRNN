import os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model_CRNN import CRNN
from SmartBedDataset_ReadOriginalTensor import SmartBedDataset_Base
import utils

use_cuda = torch.cuda.is_available()  # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU

prefix = "CRNN_01_model"
if len(sys.argv) > 1:
    prefix = sys.argv[1]
    print("Output prefix: {}".format(prefix))

suffix = ""
lf = "./" + prefix + "_testout{}.txt".format(suffix)
output_dir = "TEST_CRNN_01"
os.makedirs(output_dir, exist_ok=True)

##### basic settings #####
# time step
time_step = 10
# channel exponent to control network size
expo = 3
# add Dropout2d layer?
dropout = 0

##### create pytorch data object with dataset #####
raw_dataset = SmartBedDataset_Base(mode=SmartBedDataset_Base.TEST, time_step=time_step)
testLoader = DataLoader(raw_dataset, batch_size=1, shuffle=False)
print("Test batches: {}".format(len(testLoader)))

##### setup network #####
netG = CRNN(channelExponent=expo, dropout=dropout)
print(netG)

doLoad = "./CRNN_01_model"
if len(doLoad) > 0:
    netG.load_state_dict(torch.load(doLoad, map_location=torch.device('cpu')))
    print("Loaded model " + doLoad)
netG.to(device)

criterionMSELoss = nn.MSELoss()
MSELossTest_accum = 0.0
MSELossTest_dn_accum = 0.0
lossPer_p_accum = 0

netG.eval()

with torch.no_grad():
    for i, testdata in enumerate(testLoader, 0):
        inputs_cpu, targets_cpu = testdata
        inputs, targets = inputs_cpu.to(device), targets_cpu.to(device)

        outputs = netG(inputs)
        outputs_cpu = outputs.data.cpu().numpy()
        targets_cpu = targets.data.cpu().numpy()

        outputs = outputs.float()
        targets = targets.float()

        loss = criterionMSELoss(outputs, targets)
        MSELossViz = loss.item()
        MSELossTest_accum += MSELossViz

        lossPer_p = np.sum(np.abs(outputs_cpu - targets_cpu)) / np.sum(np.abs(targets_cpu))
        lossPer_p_accum += lossPer_p.item()

        utils.log(lf, "Test sample %d" % i)
        utils.log(lf, "    pressure:  abs. difference, ratio: %f , %f " % (
            np.sum(np.abs(outputs_cpu - targets_cpu)), lossPer_p.item()))

        targets_denormalized = raw_dataset.denormalize(targets_cpu)
        outputs_denormalized = raw_dataset.denormalize(outputs_cpu)

        targets_dn = torch.from_numpy(targets_denormalized)
        outputs_dn = torch.from_numpy(outputs_denormalized)
        MSELossTest_dn = criterionMSELoss(outputs_dn, targets_dn)
        MSELossTest_dn_accum += MSELossTest_dn.item()

        os.chdir("./TEST_CRNN_01/")
        utils.imageOut("%04d" % i, inputs_cpu[0], targets_denormalized[0], outputs_denormalized[0])
        os.chdir("../")

    utils.log(lf, "\n")
    MSELossTest_accum /= len(testLoader)
    lossPer_p_accum /= len(testLoader)
    MSELossTest_dn_accum /= len(testLoader)
    utils.log(lf, "Loss percentage of p: %f %% " % (lossPer_p_accum * 100))
    utils.log(lf, "MSE error: %f" % MSELossTest_accum)
    utils.log(lf, "Denormalized MSE error: %f" % MSELossTest_dn_accum)
    utils.log(lf, "\n")
