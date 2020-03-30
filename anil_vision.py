#!/usr/bin/env python3

import argparse
import random
import torch
import numpy as np
from tqdm import trange

import learn2learn as l2l

from utils import *
from core_functions.vision import fast_adapt, evaluate

params = {
    "ways": 5,
    "shots": 1,
    "outer_lr": 0.001,  # Outer LR should not be higher than 0.01
    "inner_lr": 0.1,  # 0.5 for 1-shot, 0.1 for 5+-shot
    "fc_neurons": 1600,
    "adapt_steps": 1,
    "meta_batch_size": 32,
    "num_iterations": 10000,
    "save_every": 1000,
    "seed": 42,
}

cl_params = {
    "adapt_steps": 1,
    "inner_lr": 0.1,
    "n_tasks": 10
}

dataset = "min"  # omni or min (omniglot / Mini ImageNet)
omni_cnn = True  # For omniglot, there is a FC and a CNN model available to choose from

cl_test = False

cuda = True

wandb = False


class Lambda(torch.nn.Module):

    def __init__(self, fn):
        super(Lambda, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class AnilVision(Experiment):

    def __init__(self):
        super(AnilVision, self).__init__(f"anil_{params['ways']}w{params['shots']}s",
                                         dataset, params, path="results/", use_wandb=wandb)

        random.seed(self.params['seed'])
        np.random.seed(self.params['seed'])
        torch.manual_seed(self.params['seed'])
        device = torch.device('cpu')
        if cuda and torch.cuda.device_count():
            print(f'Running with CUDA and device {torch.cuda.get_device_name(0)}')
            torch.cuda.manual_seed(self.params['seed'])
            device = torch.device('cuda')

        if dataset == "omni":
            train_tasks, valid_tasks, test_tasks = get_omniglot(self.params['ways'], self.params['shots'])
            input_shape = (1, 28, 28)
        elif dataset == "min":
            train_tasks, valid_tasks, test_tasks = get_mini_imagenet(self.params['ways'], self.params['shots'])
            input_shape = (3, 84, 84)
        else:
            print("Dataset not supported")
            exit(2)

        self.run(train_tasks, valid_tasks, test_tasks, input_shape, device)

    def run(self, train_tasks, valid_tasks, test_tasks, input_shape, device):

        # Create model
        features = l2l.vision.models.ConvBase(output_size=64, channels=3, max_pool=True)
        features = torch.nn.Sequential(features, Lambda(lambda x: x.view(-1, self.params['fc_neurons'])))
        features.to(device)

        head = torch.nn.Linear(self.params['fc_neurons'], self.params['ways'])
        head = l2l.algorithms.MAML(head, lr=self.params['inner_lr'])
        head.to(device)

        # Setup optimization
        all_parameters = list(features.parameters()) + list(head.parameters())
        optimizer = torch.optim.Adam(all_parameters, lr=self.params['outer_lr'])
        loss = torch.nn.CrossEntropyLoss(reduction='mean')

        self.log_model(features, device, input_shape=input_shape, name='features')  # Input shape is specific to dataset
        head_input_shape = (5, self.params['fc_neurons'])
        self.log_model(head, device, input_shape=head_input_shape, name='head')  # Input shape is specific to dataset

        t = trange(self.params['num_iterations'])
        try:
            for iteration in t:
                optimizer.zero_grad()
                meta_train_loss = 0.0
                meta_train_accuracy = 0.0
                meta_valid_loss = 0.0
                meta_valid_accuracy = 0.0
                for task in range(self.params['meta_batch_size']):
                    # Compute meta-training loss
                    learner = head.clone()
                    batch = train_tasks.sample()
                    eval_loss, eval_acc = fast_adapt(batch, learner, loss,
                                                     self.params['adapt_steps'],
                                                     self.params['shots'], self.params['ways'],
                                                     device, features=features)
                    eval_loss.backward()
                    meta_train_loss += eval_loss.item()
                    meta_train_accuracy += eval_acc.item()

                    # Compute meta-validation loss
                    learner = head.clone()
                    batch = valid_tasks.sample()
                    eval_loss, eval_acc = fast_adapt(batch, learner, loss,
                                                     self.params['adapt_steps'],
                                                     self.params['shots'], self.params['ways'],
                                                     device, features=features)
                    meta_valid_loss += eval_loss.item()
                    meta_valid_accuracy += eval_acc.item()

                meta_train_loss = meta_train_loss / self.params['meta_batch_size']
                meta_valid_loss = meta_valid_loss / self.params['meta_batch_size']
                meta_train_accuracy = meta_train_accuracy / self.params['meta_batch_size']
                meta_valid_accuracy = meta_valid_accuracy / self.params['meta_batch_size']

                metrics = {'train_loss': meta_train_loss,
                           'train_acc': meta_train_accuracy,
                           'valid_loss': meta_valid_loss,
                           'valid_acc': meta_valid_accuracy}
                t.set_postfix(metrics)
                self.log_metrics(metrics)

                # Average the accumulated gradients and optimize
                for p in all_parameters:
                    p.grad.data.mul_(1.0 / self.params['meta_batch_size'])
                optimizer.step()

                if iteration % self.params['save_every'] == 0:
                    self.save_model(features, name='/model_checkpoints/features_' + str(iteration))
                    self.save_model(head, name='/model_checkpoints/head_' + str(iteration))

        # Support safely manually interrupt training
        except KeyboardInterrupt:
            print('\nManually stopped training! Start evaluation & saving...\n')
            self.logger['manually_stopped'] = True
            self.params['num_iterations'] = iteration

        self.save_model(features, name='features')
        self.save_model(head, name='head')

        self.logger['elapsed_time'] = str(round(t.format_dict['elapsed'], 2)) + ' sec'
        # Meta-testing on unseen tasks
        self.logger['test_acc'] = evaluate(self.params, test_tasks, head, loss, device, features=features)

        self.save_logs_to_file()

        if cl_test:
            print("Running Continual Learning experiment...")
            run_cl_exp(self.model_path, head, loss, test_tasks, device,
                       self.params['ways'], self.params['shots'],
                       cl_params=cl_params, features=features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ANIL on Vision')

    parser.add_argument('--dataset', type=str, default=dataset, help='Pick a dataset')
    parser.add_argument('--ways', type=int, default=params['ways'], help='N-ways (classes)')
    parser.add_argument('--shots', type=int, default=params['shots'], help='K-shots (samples per class)')
    parser.add_argument('--seed', type=int, default=params['seed'], help='Seed')

    args = parser.parse_args()

    dataset = args.dataset
    params['ways'] = args.ways
    params['shots'] = args.shots
    params['seed'] = args.seed

    AnilVision()
