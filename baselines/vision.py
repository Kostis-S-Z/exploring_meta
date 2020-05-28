#!/usr/bin/env python3

import argparse
import random
import torch
import numpy as np
from tqdm import trange

import learn2learn as l2l

from utils import Experiment, get_omniglot, get_mini_imagenet
from core_functions.vision import accuracy, evaluate

params = {
    "ways": 5,
    "shots": 5,
    "lr": 0.001,
    "batch_size": 256,
    "num_iterations": 10000,  # 10k for Mini-ImageNet, 5k for Omniglot
    "save_every": 1000,
    "seed": 42,
}


dataset = "min"  # omni or min (omniglot / Mini ImageNet)
omni_cnn = True  # For omniglot, there is a FC and a CNN model available to choose from

log_validation = False
cl_test = False
rep_test = False

cuda = True

wandb = False


class VisionBaseline(Experiment):

    def __init__(self):
        super(VisionBaseline, self).__init__(f"bsln_{params['ways']}w{params['shots']}s",
                                             dataset, params, path="vision_results/", use_wandb=wandb)

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

        if not log_validation:
            valid_tasks = None

        self.run(train_tasks, valid_tasks, test_tasks, model, input_shape, device)

    def run(self, train_tasks, valid_tasks, test_tasks, model, input_shape, device):

        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), self.params['lr'])
        loss = torch.nn.CrossEntropyLoss(reduction='mean')

        self.log_model(model, device, input_shape=input_shape)  # Input shape is specific to dataset

        n_batch_iter = int(320 / self.params['batch_size'])
        t = trange(self.params['num_iterations'])
        try:

            for iteration in t:
                # Initialize iteration's metrics
                train_loss = 0.0
                train_accuracy = 0.0
                valid_loss = 0.0
                valid_accuracy = 0.0

                for task in range(n_batch_iter):
                    data, labels = train_tasks.sample()
                    data, labels = data.to(device), labels.to(device)

                    optimizer.zero_grad()

                    predictions = model(data)
                    batch_loss = loss(predictions, labels)
                    batch_accuracy = accuracy(predictions, labels)

                    batch_loss.backward()
                    optimizer.step()

                    train_loss += batch_loss.item()
                    train_accuracy += batch_accuracy.item()

                if valid_tasks is not None:
                    with torch.no_grad():
                        for task in range(n_batch_iter):
                            valid_data, valid_labels = train_tasks.sample()
                            valid_data, valid_labels = valid_data.to(device), valid_labels.to(device)
                            predictions = model(valid_data)
                            valid_loss += loss(predictions, valid_labels)
                            valid_accuracy += accuracy(predictions, valid_labels)

                train_loss = train_loss / n_batch_iter
                valid_loss = valid_loss / n_batch_iter
                train_accuracy = train_accuracy / n_batch_iter
                valid_accuracy = valid_accuracy / n_batch_iter

                metrics = {'train_loss': train_loss,
                           'train_acc': train_accuracy,
                           'valid_loss': valid_loss,
                           'valid_acc': valid_accuracy}
                t.set_postfix(metrics)
                self.log_metrics(metrics)

                if iteration % self.params['save_every'] == 0:
                    self.save_model_checkpoint(model, str(iteration))

        # Support safely manually interrupt training
        except KeyboardInterrupt:
            print('\nManually stopped training! Start evaluation & saving...\n')
            self.logger['manually_stopped'] = True
            self.params['num_iterations'] = iteration

        self.save_model(model)

        self.logger['elapsed_time'] = str(round(t.format_dict['elapsed'], 2)) + ' sec'
        # Testing on unseen tasks
        self.logger['test_acc'] = evaluate(self.params, test_tasks, model, loss, device)
        self.log_metrics({'test_acc': self.logger['test_acc']})
        self.save_logs_to_file()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vision baseline')

    parser.add_argument('--dataset', type=str, default=dataset, help='Pick a dataset')
    parser.add_argument('--ways', type=int, default=params['ways'], help='N-ways (classes)')
    parser.add_argument('--shots', type=int, default=params['shots'], help='K-shots (samples per class)')
    parser.add_argument('--lr', type=float, default=params['lr'], help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=params['batch_size'], help='Batch size')
    parser.add_argument('--num_iterations', type=int, default=params['num_iterations'], help='Number of epochs')
    parser.add_argument('--save_every', type=int, default=params['save_every'], help='Interval to save model')
    parser.add_argument('--seed', type=int, default=params['seed'], help='Seed')

    args = parser.parse_args()

    dataset = args.dataset
    params['ways'] = args.ways
    params['shots'] = args.shots
    params['lr'] = args.lr
    params['batch_size'] = args.batch_size
    params['num_iterations'] = args.num_iterations
    params['save_every'] = args.save_every
    params['seed'] = args.seed

    VisionBaseline()
