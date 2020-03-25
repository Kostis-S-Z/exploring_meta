#!/usr/bin/env python3

from utils.data_pre import prepare_batch


def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device, features=None):
    adapt_data, adapt_labels, eval_data, eval_labels = prepare_batch(batch, shots, ways, device, features=features)

    for step in range(adaptation_steps):
        train_error = loss(learner(adapt_data), adapt_labels)
        learner.adapt(train_error)

    predictions = learner(eval_data)
    valid_loss = loss(predictions, eval_labels)
    valid_accuracy = accuracy(predictions, eval_labels)
    return valid_loss, valid_accuracy


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def evaluate(prms, test_tasks, maml, loss, device, features=None):
    meta_test_loss = 0.0
    meta_test_accuracy = 0.0
    for task in range(prms['meta_batch_size']):
        # Compute meta-testing loss
        learner = maml.clone()
        batch = test_tasks.sample()

        eval_loss, eval_acc = fast_adapt(batch, learner, loss,
                                         prms['adapt_steps'], prms['shots'], prms['ways'],
                                         device, features=features)
        meta_test_loss += eval_loss.item()
        meta_test_accuracy += eval_acc.item()

    meta_test_accuracy = meta_test_accuracy / prms['meta_batch_size']
    print('Meta Test Accuracy', meta_test_accuracy)
    return meta_test_accuracy
