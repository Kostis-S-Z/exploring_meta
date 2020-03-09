#!/usr/bin/env python3

import argparse
import random
import torch
import numpy as np
from tqdm import trange

import learn2learn as l2l

from utils import *

# DEFAULT VALUES unless changed through command line arguments

params = dict(
    ways=5,
    shots=1,
    meta_lr=0.003,
    fast_lr=0.5,
    meta_batch_size=32,
    adaptation_steps=1,
    num_iterations=30000,
    save_every=1000,  # If you don't care about checkpoints just use an arbitrary long number e.g 1000000
    seed=42,
)

dataset = "min"  # omni or min (omniglot / Mini ImageNet)
omni_cnn = True  # For omniglot, there is a FC and a CNN model available to choose from

run_rep_test = False
run_cl_test = True

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

        # cca_results = []
        # cka1_results = []
        # cka2_results = []

        t = trange(self.params['num_iterations'])
        try:
            # sanity_batch, _ = train_tasks.sample()
            # sanity_batch = sanity_batch.to(device)
            # init_rep = self.get_rep_from_batch(maml, sanity_batch)

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

                if iteration % self.params['save_every'] == 0:
                    self.save_model_checkpoint(model, str(iteration))

                # Just a test
                # adapted_rep = self.get_rep_from_batch(maml, sanity_batch)
                # _, cca_res = get_cca_similarity(init_rep.T, adapted_rep.T, epsilon=1e-10, verbose=False)
                # cka1_k_res = linear_CKA(init_rep, adapted_rep)
                # cka2_k_res = kernel_CKA(init_rep, adapted_rep)
                # cca_results.append(cca_res)
                # cka1_results.append(cka1_k_res)
                # cka2_results.append(cka2_k_res)

        # Support safely manually interrupt training
        except KeyboardInterrupt:
            print('\nManually stopped training! Start evaluation & saving...\n')
            self.logger['manually_stopped'] = True
            self.params['num_iterations'] = iteration

        self.logger['elapsed_time'] = str(round(t.format_dict['elapsed'], 2)) + ' sec'

        # cca_results_d = dict(title="CCA Evolution",
        #                      x_legend="Train Iterations",
        #                      y_legend="CCA similarity",
        #                      y_axis=cca_results,
        #                      path=self.model_path + "/CCA_evolution.png")
        # cka1_results_d = dict(title="Linear CKA Evolution",
        #                       x_legend="Train Iterations",
        #                       y_legend="CKA similarity",
        #                       y_axis=cka1_results,
        #                       path=self.model_path + "/Linear_CKA_evolution.png")
        # cka2_results_d = dict(title="Kernel CKA Evolution",
        #                       x_legend="Train Iterations",
        #                       y_legend="CKA similarity",
        #                       y_axis=cka2_results,
        #                       path=self.model_path + "/Kernel_CKA_evolution.png")
        # plot_dict(cca_results_d, save=False)
        # plot_dict(cka1_results_d, save=False)
        # plot_dict(cka2_results_d, save=False)

        self.save_model(model)
        self.logger['test_acc'] = self.evaluate(test_tasks, maml, loss, device)

        if run_cl_test:
            print("Running Representation experiment...")
            cl_res = self.cl_test(test_tasks, maml, loss, device)
            self.logger['cl_metrics'] = cl_res

        if run_rep_test:
            print("Running Continual Learning experiment...")
            cca_res = self.representation_test(test_tasks, maml, loss, device)
            self.logger['cca'] = cca_res

        self.save_logs_to_file()

    def evaluate(self, test_tasks, maml, loss, device):
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
        return meta_test_accuracy

    def representation_test(self, test_tasks, maml, loss, device):
        """
        Measure how much the representation changes during evaluation
        """
        n_tasks = 1

        # Ignore labels
        sanity_batch, _ = test_tasks.sample()
        sanity_batch = sanity_batch.to(device)

        # An instance of the model before adaptation
        init_model = maml.clone()
        adapt_model = maml.clone()

        init_rep_sanity = self.get_rep_from_batch(init_model, sanity_batch)

        # column 0: adaptation results, column 1: init results
        acc_results = np.zeros((n_tasks, 2))
        cca_results = []
        cka1_results = []
        cka2_results = []

        for task in range(n_tasks):
            batch = test_tasks.sample()

            adapt_d, adapt_l, eval_d, eval_l = prepare_batch(batch, self.params['shots'], self.params['ways'], device)

            # Adapt the model
            for step in range(10):  # self.params['adaptation_steps']):
                train_error = loss(adapt_model(adapt_d), adapt_l)
                train_error /= len(adapt_d)
                adapt_model.adapt(train_error)

                # Evaluate the adapted model
                # a_predictions = adapt_model(eval_d)
                # a_valid_acc = accuracy(a_predictions, eval_l)

                # Evaluate the init model
                # i_predictions = init_model(eval_d)
                # i_valid_acc = accuracy(i_predictions, eval_l)

                # Get their representations
                # TODO: Which layer representation?
                # Currently getting the Fully connected
                # adapted_rep_0 = adapt_model.get_rep_i(adapt_d, 0)  # Conv1
                # adapted_rep_1 = adapt_model.get_rep_i(adapt_d, 1)  # Conv2
                # adapted_rep_2 = adapt_model.get_rep_i(adapt_d, 2)  # Conv3
                # adapted_rep_3 = adapt_model.get_rep_i(adapt_d, 3)  # Conv4
                # adapted_rep_4 = adapt_model.get_rep_i(adapt_d, 4)  # Fully connected (same as get_rep)

                # TODO: Representation on which data?
                # Option 1: adapt_d -> On the data the adaptation model was adapted
                # Option 2: eval_d -> On the data the models where evaluated
                # Option 3: different batch -> On a completely different batch of data

                adapted_rep = self.get_rep_from_batch(adapt_model, sanity_batch)
                # init_rep = self.get_rep_from_batch(init_model, sanity_batch)
                # init_rep2_sanity = self.get_rep_from_batch(init_model, sanity_batch)

                # _, sanity_cca = get_cca_similarity(init_rep_sanity.T, init_rep2_sanity.T, epsilon=1e-10, verbose=False)
                _, cca_res = get_cca_similarity(adapted_rep.T, init_rep_sanity.T, epsilon=1e-10, verbose=False)
                cka1_k_res = linear_CKA(adapted_rep, init_rep_sanity)
                cka2_k_res = kernel_CKA(adapted_rep, init_rep_sanity)

                # print(f'Sanity check: {sanity_cca} (This should always be ~1.0)')

                # acc_results[task, 0] = a_valid_acc
                # acc_results[task, 1] = i_valid_acc

                cca_results.append(cca_res)
                cka1_results.append(cka1_k_res)
                cka2_results.append(cka2_k_res)

        print("We expect that column 0 has higher values than column 1")
        print(acc_results)

        print("We expect that the values decrease over time?")
        print("CCA:", cca_results)
        print("We expect that the values decrease over time?")
        print("linear CKA:", cka1_results)
        print("We expect that the values decrease over time?")
        print("Kernerl CKA:", cka2_results)

        cca_results_d = dict(title="CCA Evolution",
                             x_legend="Inner loop steps",
                             y_legend="CCA similarity",
                             y_axis=cca_results,
                             path=self.model_path + "/inner_CCA_evolution.png")
        cka1_results_d = dict(title="Linear CKA Evolution",
                              x_legend="Inner loop steps",
                              y_legend="CKA similarity",
                              y_axis=cka1_results,
                              path=self.model_path + "/inner_Linear_CKA_evolution.png")
        cka2_results_d = dict(title="Kernel CKA Evolution",
                              x_legend="Inner loop steps",
                              y_legend="CKA similarity",
                              y_axis=cka2_results,
                              path=self.model_path + "/inner_Kernel_CKA_evolution.png")
        plot_dict(cca_results_d, save=True)
        plot_dict(cka1_results_d, save=True)
        plot_dict(cka2_results_d, save=True)

        return cca_results

    def cl_test(self, test_tasks, maml, loss, device):
        """
        Evaluate model performance in a Continual Learning setting
        """
        # For simplicity, we define as number of tasks
        n_tasks = 5  # self.params['meta_batch_size']

        option = 2
        # TODO: Should the train tasks and test tasks be the same or different?
        # Currently using Option 1
        # Option 1: Tr1 = Te1 (Exactly same samples & class)
        # Option 2: Tr1 =/= Te1 (Same class, different samples)
        # Option 3: Tr1 =///= Te1 (Different class & samples)
        # This doesn't make sense since if you define it as a 5-way problem there are only 5 classes at all times anyway
        # eval_batch = test_tasks.sample()
        # _, _, eval_d, eval_l = prepare_batch(eval_batch, self.params['shots'], self.params['ways'], device)

        # Randomly select 10 batches for training and evaluation
        tasks_pool = []
        for task_i in range(n_tasks):
            batch = test_tasks.sample()
            adapt_d, adapt_l, eval_d, eval_l = prepare_batch(batch, self.params['shots'], self.params['ways'], device)

            task = {'adapt': (adapt_d, adapt_l)}

            if option == 1:
                task['eval'] = (adapt_d, adapt_l)
            else:
                task['eval'] = (eval_d, eval_l)

            tasks_pool.append(task)

        # Matrix R NxN of accuracies in tasks j after trained on a tasks i (x_axis = test tasks, y_axis = train tasks)
        acc_matrix = np.zeros((n_tasks, n_tasks))

        # Training loop
        for i, task_i in enumerate(tasks_pool):
            adapt_i_data, adapt_i_labels = task_i['adapt']
            adapt_i_data, adapt_i_labels = adapt_i_data.to(device), adapt_i_labels.to(device)

            learner = maml.clone()
            # Adapt to task i
            for step in range(self.params['adaptation_steps']):
                train_error = loss(learner(adapt_i_data), adapt_i_labels)
                train_error /= len(adapt_i_data)
                learner.adapt(train_error)

            # Evaluation loop
            for j, task_j in enumerate(tasks_pool):
                eval_j_data, eval_j_labels = task_j['eval']
                eval_j_data, eval_j_labels = eval_j_data.to(device), eval_j_labels.to(device)

                predictions = learner(eval_j_data)
                valid_error = loss(predictions, eval_j_labels)
                valid_error /= len(eval_j_data)
                valid_accuracy_j = accuracy(predictions, eval_j_labels)

                acc_matrix[i, j] = valid_accuracy_j  # Accuracy on task j after trained on task i

        self.save_acc_matrix(acc_matrix)
        return cl_metrics(acc_matrix)

    @staticmethod
    def get_rep_from_batch(model, batch):

        representation = model.get_rep(batch)
        representation = representation.cpu().detach().numpy()

        batch_size = representation.shape[0]
        conv_neurons = representation.shape[1]
        conv_filters_1 = representation.shape[2]
        conv_filters_2 = representation.shape[3]
        representation = representation.reshape((conv_neurons * conv_filters_1 * conv_filters_2, batch_size))
        return representation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAML on Vision')

    parser.add_argument('--dataset', type=str, default='min', help='Pick a dataset')
    parser.add_argument('--ways', type=str, default='5', help='N-ways (classes)')
    parser.add_argument('--shots', type=str, default='1', help='K-shots (samples per class)')

    args = parser.parse_args()

    dataset = args.dataset
    params['ways'] = args.ways
    params['shots'] = args.shots

    MamlVision()
