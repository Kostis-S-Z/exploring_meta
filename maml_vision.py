#!/usr/bin/env python3

import argparse
import random
import torch
import numpy as np
from tqdm import trange

import learn2learn as l2l

from utils import *
from core_funtions.vision import fast_adapt

params = {
    "ways": 5,
    "shots": 1,
    "meta_lr": 0.003,
    "fast_lr": 0.5,
    "adapt_steps": 1,
    "meta_batch_size": 32,
    "num_iterations": 20000,
    "save_every": 1000,
    "seed": 42,
}

dataset = "min"  # omni or min (omniglot / Mini ImageNet)
omni_cnn = True  # For omniglot, there is a FC and a CNN model available to choose from

cl_test = False
rep_test = False

cuda = True

wandb = False


class MamlVision(Experiment):

    def __init__(self):
        super(MamlVision, self).__init__("maml", dataset, params, path="results/", use_wandb=wandb)

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
            if omni_cnn:
                model = l2l.vision.models.OmniglotCNN(self.params['ways'])
                self.params['model_type'] = 'omni_CNN'
            else:
                model = l2l.vision.models.OmniglotFC(28 ** 2, self.params['ways'])
                self.params['model_type'] = 'omni_FC'
            input_shape = (1, 28, 28)
        elif dataset == "min":
            train_tasks, valid_tasks, test_tasks = get_mini_imagenet(self.params['ways'], self.params['shots'])
            model = l2l.vision.models.MiniImagenetCNN(self.params['ways'])
            input_shape = (3, 84, 84)
        else:
            print("Dataset not supported")
            exit(2)

        self.run(train_tasks, valid_tasks, test_tasks, model, input_shape, device)

    def run(self, train_tasks, valid_tasks, test_tasks, model, input_shape, device):

        model.to(device)
        maml = l2l.algorithms.MAML(model, lr=self.params['fast_lr'], first_order=False)
        opt = torch.optim.Adam(maml.parameters(), self.params['meta_lr'])
        loss = torch.nn.CrossEntropyLoss(reduction='mean')

        self.log_model(maml, device, input_shape=input_shape)  # Input shape is specific to dataset

        t = trange(self.params['num_iterations'])
        try:

            for iteration in t:
                opt.zero_grad()
                meta_train_loss = 0.0
                meta_train_accuracy = 0.0
                meta_valid_loss = 0.0
                meta_valid_accuracy = 0.0
                for task in range(self.params['meta_batch_size']):
                    # Compute meta-training loss
                    learner = maml.clone()
                    batch = train_tasks.sample()
                    eval_loss, eval_acc = fast_adapt(batch, learner, loss,
                                                     self.params['adapt_steps'],
                                                     self.params['shots'], self.params['ways'],
                                                     device)
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
        self.logger['test_acc'] = self.evaluate(test_tasks, maml, loss, device)

        if cl_test:
            print("Running Continual Learning experiment...")
            acc_matrix, cl_res = run_cl_exp(maml, loss, test_tasks, device,
                                            self.params['ways'], self.params['shots'], self.params['adapt_steps'])
            self.save_acc_matrix(acc_matrix)
            self.logger['cl_metrics'] = cl_res

        if rep_test:
            print("Running Representation experiment...")
            self.logger['cca'] = run_rep_exp(maml, loss, test_tasks, device,
                                             self.params['ways'], self.params['shots'], self.model_path,
                                             n_tasks=1)

        self.save_logs_to_file()

    def evaluate(self, test_tasks, maml, loss, device):
        meta_test_error = 0.0
        meta_test_accuracy = 0.0
        for task in range(self.params['meta_batch_size']):
            # Compute meta-testing loss
            learner = maml.clone()
            batch = test_tasks.sample()

            evaluation_error, evaluation_accuracy = fast_adapt(batch, learner, loss,
                                                               self.params['adapt_steps'],
                                                               self.params['shots'], self.params['ways'],
                                                               device)
            meta_test_error += evaluation_error.item()
            meta_test_accuracy += evaluation_accuracy.item()

        meta_test_accuracy = meta_test_accuracy / self.params['meta_batch_size']
        print('Meta Test Accuracy', meta_test_accuracy)
        return meta_test_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAML on Vision')

    parser.add_argument('--dataset', type=str, default=dataset, help='Pick a dataset')
    parser.add_argument('--ways', type=int, default=params['ways'], help='N-ways (classes)')
    parser.add_argument('--shots', type=int, default=params['shots'], help='K-shots (samples per class)')
    parser.add_argument('--seed', type=int, default=params['seed'], help='Seed')

    args = parser.parse_args()

    dataset = args.dataset
    params['ways'] = args.ways
    params['shots'] = args.shots
    params['seed'] = args.seed

    MamlVision()
