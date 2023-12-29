import torch.onnx
from model_CRNN import CRNN

expo = 4
dropout = 0

model = CRNN(channelExponent=expo, dropout=dropout)

doLoad = "./CRNN_expo4_05_3000model"
if len(doLoad) > 0:
    model.load_state_dict(torch.load(doLoad, map_location=torch.device('cpu')))
    print("Loaded model " + doLoad)

model.eval()

dummy_input = torch.randn(1, 10, 12, 32, 64)
torch.onnx.export(model, dummy_input, "model_for_cpp.onnx")
print("done")
