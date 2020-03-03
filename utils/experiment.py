#!/usr/bin/env python3

import json
import datetime
import os

import numpy as np
import torch
from torchsummary import summary_string
import wandb


class Experiment:

    def __init__(self, algo, dataset, params, path="", wandb=False):

        self.params = params
        # Make sure all experiments have a seed
        if 'seed' in params.keys():
            seed = params['seed']
        else:
            seed = 42
            self.params.update(dict(seed=seed))

        self.logger = dict(
            config=self.params,
            date=datetime.datetime.now().strftime("%d_%m_%Hh%M"),
            model_id=str(seed) + '_' + str(np.random.randint(1, 9999)))  # Generate a unique ID based on seed + randint

        self.metrics = dict()

        if not os.path.exists(path):
            os.mkdir(path)
        # Create a unique directory for this experiment and save the model's meta-data
        self.model_path = path + algo + '_' + dataset + '_' + self.logger['date'] + '_' + self.logger['model_id']
        os.mkdir(self.model_path)
        self.save_logs_to_file()

        # Optionally, use Weights and Biases to monitor performance
        if wandb:
            self._use_wandb = True
            self._wandb = wandb.init(project="l2l", id=self.logger['model_id'])
        else:
            self._use_wandb = False

    def log_model(self, model, device, input_shape=None):
        model_info, _ = summary_string(model, input_shape, device=device)
        print(model_info)
        with open(self.model_path + "/model.summary", "w") as file:
            file.write(model_info)

        if self._use_wandb:
            wandb.watch(model)

    def log_metrics(self, metrics):
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

        if self._use_wandb:
            wandb.log(metrics)

    def save_logs_to_file(self):
        with open(self.model_path + '/metrics.json', 'w') as fp:
            json.dump(self.metrics, fp)

        with open(self.model_path + '/logger.json', 'w') as fp:
            json.dump(self.logger, fp, sort_keys=True, indent=4)

    def save_model(self, model):
        torch.save(model.state_dict(), self.model_path + '/model.pt')
        if self._use_wandb:
            torch.save(model.state_dict(), os.path.join(self._wandb.run.dir, 'model.pt'))
