import copy
import math
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import collections
from itertools import repeat
import torch.optim as optim
import numpy as np
import time

import torch
from torch import nn, Tensor
import torchvision
from torchvision.ops import StochasticDepth

from inference import load_dataset
from EfficientNet import MBConvConfig, EfficientNet_final

if torch.cuda.is_available():
    device = torch.device("cuda") 
    print("Using GPUs ", device)
else:
    device = torch.device("cpu")


def train_loop (model, data_loaders, dataset_sizes, optimizer, loss_fn, epochs, scheduler, selected_exits):
    for epoch in range(epochs):
        start_time = time.time()
        print("Epoch {}/{}".format(epoch, epochs - 1))
        print("-" * 30, flush=True)

        epoch_loss = {"train": [0.0 for i in range(len(selected_exits))], 
                    "validation": [0.0 for i in range(len(selected_exits))]}
        epoch_acc = {"train": [0.0 for i in range(len(selected_exits))], 
                    "validation": [0.0 for i in range(len(selected_exits))]}
        
        running_loss = {"train": [0.0 for i in range(len(selected_exits))], 
                        "validation": [0.0 for i in range(len(selected_exits))]}
        running_corrects = {"train": [0 for i in range(len(selected_exits))], 
                            "validation": [0 for i in range(len(selected_exits))]}

        for phase in ["train", "validation"]:
            if phase == "train":
                model.train(True)
            else:
                model.train(False)
            
            # for each batch in dataset
            for data in data_loaders[phase]:
                inputs, labels = data 
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # clear all gradients
                optimizer.zero_grad() 

                # forward pass
                outputs = model(inputs) 

                # calculate loss for all branches
                loss_list = torch.tensor(0.0, requires_grad=True, device=device)
                for out in outputs:
                    loss_list = torch.hstack((loss_list, loss_fn(out, labels)))
                    
                    
                # sum losses from all branches, no weight
                loss = torch.sum(loss_list)

                # packward pass
                if phase == "train":
                    # compute gradients
                    loss.backward()  
                    # update weights/biases
                    optimizer.step() 
                

                # update the running loss and corrects for this batch
                for i in range(len(selected_exits)):
                    _, preds = torch.max(outputs[i], 1)
                    running_loss[phase][i] += loss_list[i+1].data.item() * inputs.size(0)
                    running_corrects[phase][i] += torch.sum(preds == labels.data).item()
                    
                    
            # update the epoch loss and acc    
            epoch_loss[phase] = np.array(running_loss[phase]) / dataset_sizes[phase]
            epoch_acc[phase] =  np.array(running_corrects[phase]) / dataset_sizes[phase]

        # update learning rate when necessary
        scheduler.step()

        print('time:', np.round(time.time()-start_time, 5), "\n",
            'train_loss:', np.round(epoch_loss["train"], 4), "\n",
            'train_acc:', np.round(epoch_acc["train"], 4), "\n",
            'val_loss:', np.round(epoch_loss["validation"], 4), "\n",
            'val_acc:', np.round(epoch_acc["validation"], 4), "\n", flush=True)


def test_loop (model, data_loaders, dataset_sizes, loss_fn, selected_exits):
    print("----- test " )
    with torch.no_grad():
        start_time = time.time()
        model.eval()

        running_loss = [0.0 for i in range(len(selected_exits))]
        running_corrects = [0 for i in range(len(selected_exits))]

        for data in data_loaders['test']:
            inputs, labels = data 
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(inputs)
            
            # calculate loss for all branches
            loss_list = torch.tensor(0.0, device=device)
            for out in outputs:
                loss_list = torch.hstack((loss_list, loss_fn(out, labels)))

            # update the running loss and corrects for this batch
            for i in range(len(selected_exits)):
                _, preds = torch.max(outputs[i], 1)
                running_loss[i] += loss_list[i+1].data.item() * inputs.size(0)
                running_corrects[i] += torch.sum(preds == labels.data).item()


                
        print('time:', np.round(time.time()-start_time, 5), "\n",
            'test_loss:', np.round(np.array(running_loss)/ dataset_sizes['test'], 4), "\n",
            'test_acc:', np.round(np.array(running_corrects)/ dataset_sizes['test'], 4))
        









dataset = 'CIFAR100'
if dataset == 'CIFAR10':
    num_classes = 10
if dataset == 'CIFAR100':
    num_classes = 100

data_loaders, dataset_sizes = load_dataset (dataset, batch_size=16, image_size=224)


width_mult_b0 = 1.0
depth_mult_b0 = 1.0
inverted_residual_setting = [
    MBConvConfig(1, 3, 1, 32, 16, 1, width_mult_b0, depth_mult_b0),
    MBConvConfig(6, 3, 2, 16, 24, 2, width_mult_b0, depth_mult_b0),
    MBConvConfig(6, 5, 2, 24, 40, 2, width_mult_b0, depth_mult_b0),
    MBConvConfig(6, 3, 2, 40, 80, 3, width_mult_b0, depth_mult_b0),
    MBConvConfig(6, 5, 1, 80, 112, 3, width_mult_b0, depth_mult_b0),
    MBConvConfig(6, 5, 2, 112, 192, 4, width_mult_b0, depth_mult_b0),
    MBConvConfig(6, 3, 1, 192, 320, 1, width_mult_b0, depth_mult_b0),
]


selected_exits = np.array([1, 2, 3, 4, 5, 6, 7, 8])
model = EfficientNet_final(inverted_residual_setting, dropout=0.2, stochastic_depth_prob=0.2, num_classes=num_classes, selected_exits=selected_exits).to(device)

### loading pytorch weights on the new efficientnet model, 
# by changing keys and ignoring exit branches
the_dict_mine = model.state_dict()
the_dict_theirs = torch.load('weights/efficientnet_b0_rwightman-3dd342df.pth')
new_dict = {}
for key_mine, key_thiers in zip(the_dict_mine, the_dict_theirs):
    if 'exit' not in key_mine:
        new_dict[key_mine] = the_dict_theirs[key_thiers]
model.load_state_dict(new_dict, strict=False)




epochs = 80
start_lr = 0.001

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=start_lr, momentum=0.9)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 70], gamma=0.1)

train_loop (model, data_loaders, dataset_sizes, optimizer, loss_fn, epochs, scheduler, selected_exits)
test_loop (model, data_loaders, dataset_sizes, loss_fn, selected_exits)




### save model, method1, state_dict only, not the entire model
na = str(list(selected_exits)).replace(" ", "")
torch.save(model.state_dict(),f"weights/EfficientNetB0_{dataset}_state_dict_final{na}.pt")

### save model, method2, entire model
torch.save(model,f"weights/EfficientNetB0_{dataset}_entire_model_final{na}.pt")

