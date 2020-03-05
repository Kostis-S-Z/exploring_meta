#!/usr/bin/env python3

import random
import torch
import numpy as np
from tqdm import trange

import learn2learn as l2l

from utils import *

params = dict(
    ways=5,
    shots=1,
    meta_lr=0.003,
    fast_lr=0.5,
    meta_batch_size=32,
    adaptation_steps=1,
    num_iterations=5,
    seed=42,
)

dataset = "min"  # omni or min (omniglot / Mini ImageNet)
omni_cnn = True  # For omniglot, there is a FC and a CNN model available to choose from

rep_test = False
cl_test = True

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
                meta_train_error = 0.0
                meta_train_accuracy = 0.0
                meta_valid_error = 0.0
                meta_valid_accuracy = 0.0
                for task in range(self.params['meta_batch_size']):
                    # Compute meta-training loss
                    learner = maml.clone()
                    batch = train_tasks.sample()
                    evaluation_error, evaluation_accuracy = maml_fast_adapt(batch, learner, loss,
                                                                            self.params['adaptation_steps'],
                                                                            self.params['shots'], self.params['ways'],
                                                                            device)
                    evaluation_error.backward()
                    meta_train_error += evaluation_error.item()
                    meta_train_accuracy += evaluation_accuracy.item()

                    # Compute meta-validation loss
                    learner = maml.clone()
                    batch = valid_tasks.sample()
                    evaluation_error, evaluation_accuracy = maml_fast_adapt(batch, learner, loss,
                                                                            self.params['adaptation_steps'],
                                                                            self.params['shots'], self.params['ways'],
                                                                            device)
                    meta_valid_error += evaluation_error.item()
                    meta_valid_accuracy += evaluation_accuracy.item()

                # Print some metrics
                meta_train_accuracy = meta_train_accuracy / self.params['meta_batch_size']
                meta_valid_accuracy = meta_valid_accuracy / self.params['meta_batch_size']

                metrics = {'train_acc': meta_train_accuracy,
                           'valid_acc': meta_valid_accuracy}
                t.set_postfix(metrics)
                self.log_metrics(metrics)

                # Average the accumulated gradients and optimize
                for p in maml.parameters():
                    p.grad.data.mul_(1.0 / self.params['meta_batch_size'])
                opt.step()

                if iteration % 1000 == 0:
                    self.save_model(model, name="model_i_" + str(iteration))
        # Support safely manually interrupt training
        except KeyboardInterrupt:
            print('\nManually stopped training! Start evaluation & saving...\n')
            self.logger['manually_stopped'] = True
            self.params['num_iterations'] = iteration

        self.save_model(model)

        meta_test_error = 0.0
        meta_test_accuracy = 0.0
        for task in range(self.params['meta_batch_size']):
            # Compute meta-testing loss
            learner = maml.clone()
            batch = test_tasks.sample()

            evaluation_error, evaluation_accuracy = maml_fast_adapt(batch, learner, loss,
                                                                    self.params['adaptation_steps'],
                                                                    self.params['shots'], self.params['ways'],
                                                                    device)
            meta_test_error += evaluation_error.item()
            meta_test_accuracy += evaluation_accuracy.item()

        meta_test_accuracy = meta_test_accuracy / self.params['meta_batch_size']
        print('Meta Test Accuracy', meta_test_accuracy)

        self.logger['elapsed_time'] = str(round(t.format_dict['elapsed'], 2)) + ' sec'
        self.logger['test_acc'] = meta_test_accuracy

        if cl_test:
            cl_res = self.calc_cl(test_tasks, maml, loss, device)
            self.logger['cl_metrics'] = cl_res

        if rep_test:
            cca_res = self.representation_test(test_tasks, learner, maml, loss, device)
            self.logger['cca'] = cca_res

        self.save_logs_to_file()

    def representation_test(self, test_rep_tasks, learner, maml, loss, device):
        # TEST REPRESENTATION
        rep_ways = 5
        rep_shots = 1
        n_samples = rep_ways * rep_shots

        # if dataset == "omni":
        #     _, _, test_rep_tasks = get_omniglot(rep_ways, rep_shots)
        # elif dataset == "min":
        #     _, _, test_rep_tasks = get_mini_imagenet(rep_ways, rep_shots)
        # else:
        #     print("Dataset not supported")
        #     exit(2)
        #
        test_rep_batch, _, _, _ = prepare_batch(test_rep_tasks.sample(), rep_ways, rep_shots, device)

        init_net_rep = learner.get_rep(test_rep_batch)  # Trained representation before meta-testing
        init_rep = init_net_rep.cpu().detach().numpy()
        init_rep = init_rep.reshape((n_samples * 5 * 5, self.params['meta_batch_size']))

        meta_test_error = 0.0
        meta_test_accuracy = 0.0
        for task in range(self.params['meta_batch_size']):
            # Compute meta-testing loss
            learner = maml.clone()
            batch = test_rep_tasks.sample()

            prev_net_rep = learner.get_rep(test_rep_batch)  # Get rep before adaptation

            evaluation_error, evaluation_accuracy = maml_fast_adapt(batch, learner, loss,
                                                                    self.params['adaptation_steps'],
                                                                    self.params['shots'],
                                                                    self.params['ways'],
                                                                    device)
            meta_test_error += evaluation_error.item()
            meta_test_accuracy += evaluation_accuracy.item()

            new_net_rep = learner.get_rep(test_rep_batch)  # Get rep after adaptation

            prev_rep = prev_net_rep.cpu().detach().numpy()
            new_rep = new_net_rep.cpu().detach().numpy()

            prev_rep = prev_rep.reshape((n_samples * 5 * 5, self.params['meta_batch_size']))
            new_rep = new_rep.reshape((n_samples * 5 * 5, self.params['meta_batch_size']))

            # cca_res = get_cca_similarity(prev_rep.T, new_rep.T, epsilon=1e-10, verbose=False)
            # cka_l_res = linear_CKA(prev_rep.T, new_rep.T)
            # cka_k_res = kernel_CKA(prev_rep.T, new_rep.T)

            # print('CCA: {:.4f}'.format(np.mean(cca_res["cca_coef1"])))
            # print('Linear CKA: {:.4f}'.format(cka_l_res))
            # print('Kernel CKA: {:.4f}'.format(cka_k_res))

        final_cca_res = get_cca_similarity(init_rep.T, new_rep.T, epsilon=1e-10, verbose=False)
        # final_cka_l_res = linear_CKA(init_rep, new_rep)
        # final_cka_k_res = kernel_CKA(init_rep, new_rep)

        print('Final results between representations of shape', init_rep.shape)
        print('     CCA: {:.4f}'.format(np.mean(final_cca_res["cca_coef1"])))
        # print('     Linear CKA: {:.4f}'.format(final_cka_l_res))
        # print('     Kernel CKA: {:.4f}'.format(final_cka_k_res))

        return np.mean(final_cca_res["cca_coef1"])

    def calc_cl(self, test_tasks, maml, loss, device):

        # For simplicity, we define as number of tasks
        n = 5  # self.params['meta_batch_size']

        # TODO: Should the train tasks and test tasks be the same or different?
        # Currently using Option 1
        # Option 1: Tr1 = Te1 (test task 1 is the same samples and class as train task 1)
        # Option 2: Tr1 =/= Te1 but same class (test task 1 is different samples but same class as train task 1)
        # Option 3: Tr1 =///= Te1 (test task 1 is completely different class from train task 1)

        # Randomly select 10 batches for training and evaluation
        tasks_pool = []
        for task in range(n):
            batch = test_tasks.sample()
            tasks_pool.append(batch)

        # Matrix R NxN of accuracies in tasks j after trained on a tasks i (x_axis = test tasks, y_axis = train tasks)
        acc_matrix = np.zeros((n, n))

        # Training loop
        for i, task_i in enumerate(tasks_pool):
            adapt_i_data, adapt_i_labels = task_i
            adapt_i_data, adapt_i_labels = adapt_i_data.to(device), adapt_i_labels.to(device)

            learner = maml.clone()
            # Adapt to task i
            for step in range(self.params['adaptation_steps']):
                train_error = loss(learner(adapt_i_data), adapt_i_labels)
                train_error /= len(adapt_i_data)
                learner.adapt(train_error)

            # Evaluation loop
            for j, task_j in enumerate(tasks_pool):
                eval_j_data, eval_j_labels = task_j
                eval_j_data, eval_j_labels = eval_j_data.to(device), eval_j_labels.to(device)

                predictions = learner(eval_j_data)
                valid_error = loss(predictions, eval_j_labels)
                valid_error /= len(eval_j_data)
                valid_accuracy_j = accuracy(predictions, eval_j_labels)

                acc_matrix[i, j] = valid_accuracy_j  # Accuracy on task j after trained on task i

        self.save_acc_matrix(acc_matrix)
        return cl_metrics(acc_matrix)


if __name__ == '__main__':
    MamlVision()
