import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from tqdm import tqdm
import os
from .nn_pruning_utils import load_model_from_checkpoint, val, create_dataloaders
import numpy as np
import gc

# count number of parameter and constraint params
def count_parameters(model):
    count = 0
    param_size_list = []
    for name, param in model.named_parameters():
        if 'weight' in name and 'conv' in name:
            count += 1
            param_size_list.append(param.data.numel())
    return count, np.array(param_size_list)


def global_pruning_per_layer(pruning_rate, model_name: str):
    model = load_model_from_checkpoint(model_name)
    for name, param in model.named_parameters():
        if 'weight' in name and 'conv' in name:
            tensor = param.data.cpu().numpy()
            alive = tensor.reshape(-1)  # flattened array of nonzero values
            percentile_value = np.percentile(abs(alive), pruning_rate*100)

            weight_dev = param.device
            dtype = param.data.dtype
            new_mask = np.where(abs(tensor) < percentile_value, 0.0, 1.0)

            param.data = torch.from_numpy(tensor * new_mask).to(dtype).to(weight_dev)
    return model


def global_pruning(pruning_rate, model_name: str):
    model = load_model_from_checkpoint(model_name)
    all_alive = []
    optimal = []
    for name, param in model.named_parameters():
        if 'weight' in name and 'conv' in name:
            tensor = param.data.cpu().numpy()
            alive = tensor.reshape(-1)
            all_alive.append(alive)

    all_alive = np.concatenate(all_alive)
    percentile_value = np.percentile(abs(all_alive), pruning_rate*100.0)

    for name, param in model.named_parameters():
        if 'weight' in name and 'conv' in name:
            tensor = param.data.cpu().numpy()
            weight_dev = param.device
            dtype = param.data.dtype
            new_mask = np.where(abs(tensor) < percentile_value, 0.0, 1.0)

            param.data = torch.from_numpy(tensor * new_mask).to(dtype).to(weight_dev)

            # calculate optimal
            alive = tensor.reshape(-1)
            layer_pruning_rate = sum(np.abs(alive) < percentile_value) / float(len(alive))
            optimal.append(layer_pruning_rate)
    return model, np.array(optimal)


def bo_pruning(pruning_rate_list, model_name: str):
    model = load_model_from_checkpoint(model_name)
    dims, _ = count_parameters(model)
    assert len(pruning_rate_list) == dims, "Invalid pruning rate list, too less or too many"
    assert max(pruning_rate_list) <= 1.0, "Max rate larger than 1.0, invalid"
    assert min(pruning_rate_list) >= 0.0, "min rate small than 0.0, invalid"
    i = 0

    for name, param in model.named_parameters():
        if 'weight' in name and 'conv' in name:
            tensor = param.data.cpu().numpy()
            alive = tensor.reshape(-1)
            percentile_value = np.percentile(abs(alive), pruning_rate_list[i]*100.0)

            weight_dev = param.device
            dtype = param.data.dtype
            new_mask = np.where(abs(tensor) < percentile_value, 0.0, 1.0)

            param.data = torch.from_numpy(tensor * new_mask).to(dtype).to(weight_dev)
            i += 1

    return model


class NNPruning:
    def __init__(self, model_name: str):
        model = load_model_from_checkpoint(model_name)
        self.dims, self.constraint = count_parameters(model)
        self.device = torch.device('cuda')
        del model

        self.lb = np.zeros(self.dims, )
        self.ub = np.ones(self.dims, )

    def __call__(self, x):
        x = np.array(x)
        if x.ndim == 0:
            x = np.expand_dims(x, 0)
        assert x.ndim == 1

        val_dataloader = create_dataloaders(100, 64)["val"]
        bo_pruned_model = bo_pruning(x, model_name)
        _, val_avg_acc = val(bo_pruned_model, val_dataloader, device)
        val_avg_acc = val_avg_acc.detach().cpu()
        del bo_pruned_model
        return val_avg_acc.item()

    def get_constraint_param(self):
        return self.constraint


# higher the pruning rate, the more to pruning
class NNPruningConstraint:
    def __init__(self, model_name: str, avg_pruning_rate: float):
        model = load_model_from_checkpoint(model_name)
        self.dims, self.constraint = count_parameters(model)
        self.device = torch.device('cuda')
        self.model_name = model_name
        self.avg_pruning_rate = avg_pruning_rate
        del model

        self.lb = np.zeros(self.dims, )
        self.ub = np.ones(self.dims, )

    def __call__(self, x):
        x = np.array(x)
        if x.ndim == 0:
            x = np.expand_dims(x, 0)
        assert x.ndim == 1

        # scale from [0, 1] to [0, 0.95] for densenet201
        if self.model_name == "densenet201":
            x = x * 0.9

        val_dataloader = create_dataloaders(100, 64)["val"]
        bo_pruned_model = bo_pruning(x, self.model_name)
        _, val_avg_acc = val(bo_pruned_model, val_dataloader, self.device)

        # get constraint penalty
        N = np.sum(self.constraint)
        bo_alive = N - np.sum(x*self.constraint)
        global_alive = N - N * self.avg_pruning_rate
        penalty = np.clip(bo_alive-global_alive, a_min=0.0, a_max=None) / N

        # 5.0, 8.0, 15.0

        f = val_avg_acc.item() - 8.0*penalty

        return f

    def get_constraint_param(self):
        return self.constraint

# x: rate(N, ), params per layers (N, ) ----> x dot param == sum(param) * global rate

# x (N,)   ---> [0, 1]
# x (N,)   x[N-1, ] = 1 N < 0

if __name__ == '__main__':
    info = torch.cuda.mem_get_info()[0] // 1024**3
    gc.collect()
    torch.cuda.empty_cache()
    model_name = "densenet201"
    model = load_model_from_checkpoint(model_name)
    global_pruned_model, optimal = global_pruning(0.5, model_name)
    layer_pruned_model = global_pruning_per_layer(0.5, model_name)
    # bo_pruned_model = bo_pruning(optimal, model_name)

    device = torch.device('cuda')
    val_dataloader = create_dataloaders(100, 64)["val"]
    """
    _, val_avg_acc = val(model, val_dataloader, device)
    print(f"{model_name}(Without pruning) ----- Val acc: {val_avg_acc}")
    """
    _, val_avg_acc = val(layer_pruned_model, val_dataloader, device)
    print(f"{model_name}(Layer prune rate) ----- Val acc: {val_avg_acc}")

   #  _, val_avg_acc = val(global_pruned_model, val_dataloader, device)
    print(f"{model_name}(Global prune rate) ----- Val acc: {val_avg_acc}")



    # bo pruning
    # benchmark = NNPruningConstraint(model_name, 0.5)
    # val_avg_acc = benchmark(optimal)
    # val_avg_acc = benchmark(optimal)
    # print(f"{model_name}(BO pruning rate test) ----- Val acc: {val_avg_acc}")
