import torch
import torch_xla.core.xla_model as xm
import time
import pickle

import numpy as np
import torchvision.models as models

import sys

class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.linear1 = torch.nn.Linear(100, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 10)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

print('if stalls here, check tpu ip address')
device = xm.xla_device()

# LeNet
#model = LeNet()
#model.to(device)

#layer1 = pickle.load(open("fc_1.p", "rb"))
#layer2 = pickle.load(open("fc_2.p", "rb"))
#layer3 = pickle.load(open("fc_3.p", "rb"))

#print(len(layer1), len(layer1[0]), len(layer2), len(layer2[0]), len(layer3), len(layer3[0]))

# GPT-2
from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2Model.from_pretrained('gpt2').to(device)

import random
import string

def get_random_string(length):
    # With combination of lower and upper case
    result_str = ''.join(random.choice(string.ascii_letters) for i in range(length))
    # print random string
    return result_str

model = models.alexnet(pretrained=True).to(device)

# get model size
param_size = 0
for param in model.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()
size_all_mb = (param_size + buffer_size) / 1024**2
print('model size: {:.3f}MB'.format(size_all_mb))

dummy_input = torch.randn(1, 3,224,224,dtype=torch.float).to(device)
# dummy input for GPT-2
# dummy_input = tokenizer(get_random_string(50), return_tensors='pt').to(device)
repetitions = 300
timings=np.zeros((repetitions,1))
#GPU-WARM-UP
for i in range(100):
   _ = model(dummy_input)
   # GPT-2 warm-up
   # _ = model(**dummy_input)
# MEASURE PERFORMANCE
with torch.no_grad():
  for rep in range(repetitions):
     dummy_input = torch.randn(1, 3,224,224,dtype=torch.float).to(device)
     # dummy_input = tokenizer(get_random_string(50), return_tensors='pt').to(device)
     #start = time.perfcounter()
     start = time.time()
     _ = model(dummy_input)
     #nference_time = time.perfcounter() - start
     inference_time = time.time() - start
     #_ = model(**dummy_input)
     # WAIT FOR GPU SYNC
     timings[rep] = inference_time
mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
#print(timings)
print(mean_syn)
