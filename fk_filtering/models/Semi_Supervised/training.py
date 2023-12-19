import torch as pt
from torch import nn
from ...simulation import Differentiable_Particle_Filter
from copy import copy
from tqdm import tqdm
from typing import Callable, Iterable
from matplotlib import pyplot as plt
import numpy as np
from ...results import Reporter
from copy import deepcopy
from graphviz import Digraph
from torch.autograd import Variable, Function
from ...loss import Loss

device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

def test(
        DPF: Differentiable_Particle_Filter, 
        statistics: Iterable[Reporter], 
        T: int, 
        data: pt.utils.data.Dataset, 
        batch_size: int, 
        ):
    
    DPF.to(device=device)
    DPF.eval()
    try:
        test_loader = pt.utils.data.DataLoader(data, batch_size, shuffle=True, collate_fn=data.collate)
    except:
        test_loader = data
    ESS_ratio = DPF.ESS_threshold / DPF.n_particles
    DPF.n_particles = 2000
    DPF.ESS_threshold = 2000 * ESS_ratio
    with pt.inference_mode():
        for i, simulated_object in enumerate(tqdm(test_loader)):
            statistics_ = [copy(statistic) for statistic in statistics]
            DPF(simulated_object, T, statistics_)
            pos_error = pt.mean(pt.sqrt(pt.mean((statistics_[0].results[:,:,:2] - simulated_object.state[:,:T+1,:2])**2, dim = 2)), dim = 1)
            d_a = pt.abs(statistics_[0].results[:,:,2] - simulated_object.state[:,:T+1,2])
            angle_error = pt.mean(pt.minimum(d_a, 2*pt.pi - d_a), dim = 1)
            try:
                total_pos_error = pt.concat(pos_error, pos_error.cpu().detach().numpy())
                total_angle_error = pt.concat(angle_error, angle_error.cpu().detach().numpy())
            except:
                total_pos_error = pos_error.cpu().detach().numpy()
                total_angle_error = angle_error.cpu().detach().numpy()
            if i == 0:
                plt.plot(statistics_[0].results[0, :20, 0].detach().cpu().numpy(), statistics_[0].results[0, :20, 1].detach().cpu().numpy())
                plt.plot(simulated_object.state[0, :20, 0].detach().cpu().numpy(), simulated_object.state[0, :20, 1].detach().cpu().numpy())
                plt.xlim((0, 1000))
                plt.ylim((0, 500))
                plt.show()
        
    print(f'Mean postion error: {np.mean(total_pos_error)}')
    print(f'Worst postion error: {np.max(total_pos_error)}')
    print(f'Best postion error: {np.min(total_pos_error)}')
    print(f'Std postion error: {np.std(total_pos_error)}')
    print(f'Mean angle error: {np.mean(total_angle_error)}')
    print(f'Worst angle error: {np.max(total_angle_error)}')
    print(f'Best angle error: {np.min(total_angle_error)}')
    print(f'Std angle error: {np.std(total_angle_error)}')
    return np.mean(total_pos_error)


def e2e_train(
        DPF: Differentiable_Particle_Filter, 
        loss_fn: Loss, 
        statistics: Iterable[Reporter], 
        T: int, 
        data: pt.utils.data.Dataset, 
        batch_size: int,  
        epochs: int, 
        lr: float,
        labeled_ratio: float,
        verbose = True
        ):
    
    DPF.to(device=device)
    train_set, test_set, f_test_set = pt.utils.data.random_split(data, [0.45, 0.05, 0.5])
    train = pt.utils.data.DataLoader(train_set, batch_size, shuffle=False, collate_fn=data.collate, drop_last=True)
    valid = pt.utils.data.DataLoader(test_set, len(test_set), shuffle=False, collate_fn=data.collate, drop_last=True)
    f_test = pt.utils.data.DataLoader(f_test_set, len(f_test_set), collate_fn=data.collate, drop_last=True)
    train_loss = np.zeros(len(train)*epochs)
    angle_error = np.zeros(epochs)
    pos_error = np.zeros(epochs)
    data_length = len(train)*batch_size*(T+1)
    mask_1 = pt.ones(int(data_length*labeled_ratio), device=device)
    mask_2 = pt.zeros(data_length - len(mask_1), device=device)
    mask = pt.concat((mask_1, mask_2))[pt.randperm(data_length)].reshape((len(train), batch_size, T+1))
    last_e = 0
    try:
        opt = pt.optim.AdamW(DPF.parameters(), lr=lr)
    except:
        return test(DPF, loss_fn, statistics, T, f_test, len(f_test))
    min_valid_loss = pt.inf
    
    for epoch in range(epochs):
        DPF.train()
        if verbose:
            train_it = enumerate(tqdm(train))
        else:
            train_it = enumerate(train)

        for b, simulated_object in train_it:
            opt.zero_grad()
            statistics_ = [copy(statistic) for statistic in statistics]
            DPF(simulated_object, T, statistics_)
            loss = loss_fn([(mask[b], statistics_[0], simulated_object),[statistics_[1]]])
            loss.backward()
            opt.step()
            train_loss[b + len(train)*epoch] = loss.item()

        DPF.eval()

        with pt.inference_mode():
            
            for simulated_object in valid:
                statistics_ = [copy(statistic) for statistic in statistics]
                DPF(simulated_object, T, statistics_)
                pos_error[epoch] += pt.mean(pt.sqrt(pt.sum((statistics_[0].results[:,:,:2] - simulated_object.state[:,:T+1,:2])**2, dim = 2))).item()
                d_a = pt.abs(statistics_[0].results[:,:,2] - simulated_object.state[:,:T+1,2])
                angle_error[epoch] += pt.mean(pt.minimum(d_a, 2*pt.pi - d_a)).item()
            pos_error[epoch] /= len(valid)
            angle_error[epoch] /= len(valid)

        if pos_error[epoch] < min_valid_loss:
            last_e = epoch
            min_valid_loss = pos_error[epoch]
            best_dict = deepcopy(DPF.state_dict())

        if verbose:
            print(f'Epoch {epoch}:')
            print(f'Train loss: {np.mean(train_loss[epoch*len(train):(epoch+1)*len(train)])}')
            print(f'Distance Error: {pos_error[epoch]}')
            print(f'Angle Error: {angle_error[epoch]} \n')
 
        #if epoch - last_e > 10 and epoch > 100:
            #break
    
    if verbose:
        plt.plot(train_loss)
        plt.title('Train Loss')
        plt.xlabel('Batch')
        plt.ylabel('Train loss')
        plt.show()
        plt.plot(pos_error)
        plt.title('Validation Position Error')
        plt.xlabel('Epoch')
        plt.ylabel('Position Error')
        plt.show()
        plt.plot(angle_error)
        plt.title('Validation Angle Error')
        plt.xlabel('Epoch')
        plt.ylabel('Angle Error')
        plt.show()

    DPF.load_state_dict(best_dict)
    return test(DPF, statistics, T, f_test, 100)