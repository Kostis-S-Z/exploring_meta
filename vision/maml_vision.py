#!/usr/bin/env python3

import argparse
import random
import torch
import numpy as np
from tqdm import trange

from learn2learn.algorithms import MAML

from utils import *
from core_functions.vision import fast_adapt, evaluate
from core_functions.vision_models import OmniglotCNN, MiniImagenetCNN

params = {
    "ways": 5,
    "shots": 1,
    "outer_lr": 0.003,
    "inner_lr": 0.5,
    "adapt_steps": 1,
    "meta_batch_size": 32,
    "num_iterations": 10000,  # 10k for Mini-ImageNet, 5k for Omniglot
    "save_every": 1000,
    "seed": 42,
}

cl_params = {
    "adapt_steps": 1,
    "inner_lr": 0.1,
    "n_tasks": 10
}

rep_params = {
    "adapt_steps": 1,
    "inner_lr": 0.1,
    "n_tasks": 5,
    "layers": [4]
}

dataset = "min"  # omni or min (omniglot / Mini ImageNet)
omni_cnn = True  # For omniglot, there is a FC and a CNN model available to choose from

cuda = True

wandb = True


class MamlVision(Experiment):

    def __init__(self):
        super(MamlVision, self).__init__(f"maml_{params['ways']}w{params['shots']}s",
                                         dataset, params, path="results/", use_wandb=wandb)

        # Initialize seeds & devices
        random.seed(self.params['seed'])
        np.random.seed(self.params['seed'])
        torch.manual_seed(self.params['seed'])
        device = torch.device('cpu')
        if cuda and torch.cuda.device_count():
            print(f'Running with CUDA and device {torch.cuda.get_device_name(0)}')
            torch.cuda.manual_seed(self.params['seed'])
            device = torch.device('cuda')

        # Fetch data as tasks
        if dataset == "omni":
            train_tasks, valid_tasks, test_tasks = get_omniglot(self.params['ways'], self.params['shots'])
            if omni_cnn:
                model = OmniglotCNN(self.params['ways'])
                self.params['model_type'] = 'omni_CNN'
            input_shape = (1, 28, 28)
        elif dataset == "min":
            train_tasks, valid_tasks, test_tasks = get_mini_imagenet(self.params['ways'], self.params['shots'])
            model = MiniImagenetCNN(self.params['ways'])
            input_shape = (3, 84, 84)
        else:
            print("Dataset not supported")
            exit(2)

        self.run(train_tasks, valid_tasks, test_tasks, model, input_shape, device)

    def run(self, train_tasks, valid_tasks, test_tasks, model, input_shape, device):

        model.to(device)
        maml = MAML(model, lr=self.params['inner_lr'], first_order=False)
        opt = torch.optim.Adam(maml.parameters(), self.params['outer_lr'])
        loss = torch.nn.CrossEntropyLoss(reduction='mean')

        self.log_model(maml, device, input_shape=input_shape)  # Input shape is specific to dataset

        t = trange(self.params['num_iterations'])
        try:

            for iteration in t:
                # Clear the gradients after successfully back-propagating through the whole network
                opt.zero_grad()
                # Initialize iteration's metrics
                meta_train_loss = 0.0
                meta_train_accuracy = 0.0
                meta_valid_loss = 0.0
                meta_valid_accuracy = 0.0
                # Inner (Adaptation) loop
                for task in range(self.params['meta_batch_size']):
                    # Compute meta-training loss
                    learner = maml.clone()
                    batch = train_tasks.sample()
                    eval_loss, eval_acc = fast_adapt(batch, learner, loss,
                                                     self.params['adapt_steps'],
                                                     self.params['shots'], self.params['ways'],
                                                     device)

                    # Calculate the gradients of the now updated parameters of the model using the evaluation loss!
                    eval_loss.backward()
                    meta_train_loss += eval_loss.item()
                    meta_train_accuracy += eval_acc.item()

                    # Compute meta-validation loss
                    learner = maml.clone()
                    batch = valid_tasks.sample()
                    eval_loss, eval_acc = fast_adapt(batch, learner, loss,
                                                     self.params['adapt_steps'],
                                                     self.params['shots'], self.params['ways'],
                                                     device)
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
                for p in maml.parameters():
                    p.grad.data.mul_(1.0 / self.params['meta_batch_size'])
                opt.step()

                if iteration % self.params['save_every'] == 0:
                    self.save_model_checkpoint(model, str(iteration))

        # Support safely manually interrupt training
        except KeyboardInterrupt:
            print('\nManually stopped training! Start evaluation & saving...\n')
            self.logger['manually_stopped'] = True
            self.params['num_iterations'] = iteration

        self.save_model(model)

        self.logger['elapsed_time'] = str(round(t.format_dict['elapsed'], 2)) + ' sec'
        # Meta-testing on unseen tasks
        self.logger['test_acc'] = evaluate(self.params, test_tasks, maml, loss, device)
        self.log_metrics({'test_acc': self.logger['test_acc']})
        self.save_logs_to_file()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAML on Vision')

    parser.add_argument('--dataset', type=str, default=dataset, help='Pick a dataset')
    parser.add_argument('--ways', type=int, default=params['ways'], help='N-ways (classes)')
    parser.add_argument('--shots', type=int, default=params['shots'], help='K-shots (samples per class)')

    parser.add_argument('--outer_lr', type=float, default=params['outer_lr'], help='Outer lr')
    parser.add_argument('--inner_lr', type=float, default=params['inner_lr'], help='Inner lr')
    parser.add_argument('--adapt_steps', type=int, default=params['adapt_steps'], help='Adaptation steps in inner loop')
    parser.add_argument('--meta_batch_size', type=int, default=params['meta_batch_size'], help='Batch size')
    parser.add_argument('--num_iterations', type=int, default=params['num_iterations'], help='Number of epochs')
    parser.add_argument('--save_every', type=int, default=params['save_every'], help='Interval to save model')

    parser.add_argument('--seed', type=int, default=params['seed'], help='Seed')

    args = parser.parse_args()

    dataset = args.dataset
    params['ways'] = args.ways
    params['shots'] = args.shots
    params['outer_lr'] = args.outer_lr
    params['inner_lr'] = args.inner_lr
    params['adapt_steps'] = args.adapt_steps
    params['meta_batch_size'] = args.meta_batch_size
    params['num_iterations'] = args.num_iterations
    params['save_every'] = args.save_every

    params['seed'] = args.seed

    MamlVision()
