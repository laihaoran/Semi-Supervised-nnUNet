#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch
from torch import nn
import torch.nn.functional as F


softmax_helper = lambda x: F.softmax(x, 1)


def ouput_to_softmax_combine(data1, data2):
    if isinstance(data1, list):
        data = [(softmax_helper(i) + softmax_helper(j)) / 2 for i, j in zip(data1, data2)]
    elif isinstance(data1, tuple):
        data = tuple([(softmax_helper(i) + softmax_helper(j)) / 2  for i, j in zip(data1, data2)])
    else:
        data = (softmax_helper(data1) + softmax_helper(data2)) / 2
    return data


def ouput_to_mask(data):
    if isinstance(data, list):
        data = [softmax_helper(i).argmax(dim=1).unsqueeze(1) for i in data]
    elif isinstance(data, tuple):
        data = [softmax_helper(i).argmax(dim=1).unsqueeze(1) for i in data]
    else:
        data = softmax_helper(data).argmax(dim=1).unsqueeze(1)
    return data


def cut_mix(mix_mask, data1, data2):
    if isinstance(data1, list):
        data = [j * (1 - i) + k * i for i,j, k in zip(mix_mask, data1, data2)]
    elif isinstance(data1, tuple):
        data = tuple([j * (1 - i) + k * i for i,j, k in zip(mix_mask, data1, data2)])
    else:
        data = data1 * (1 - mix_mask) + data2 * mix_mask
    return data