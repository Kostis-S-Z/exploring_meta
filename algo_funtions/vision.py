#!/usr/bin/env python3

from utils.data_pre import prepare_batch


def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device, features=None):
    adapt_data, adapt_labels, eval_data, eval_labels = prepare_batch(batch, shots, ways, device, features=features)

    for step in range(adaptation_steps):
        train_error = loss(learner(adapt_data), adapt_labels)
        learner.adapt(train_error)

    predictions = learner(eval_data)
    valid_error = loss(predictions, eval_labels)
    valid_accuracy = accuracy(predictions, eval_labels)
    return valid_error, valid_accuracy


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)
