# Walkthrough of the code for MAML & ANIL in Vision experiments


## __Definitions__
- Iterations = epochs = number of updates
- Network = model ~ policy
- Inner lr (α): step value for inner loop adaptation of MAML / ANIL
- Outer lr (β): step value for outer loop optimization of the the model

## __Data__

### Omniglot
1623 classes (characters). 20 samples per class.

Train set: 0-1100

Valid set: 1100-1200

Test set: 1200-1623

Random rotations of 90 / 180 / 270 degrees.

### Mini-ImageNet
100 classes. 600 samples per class.

84x84x3 (RGB) images

Train set: 64 classes

Valid set: 16 classes

Test set: 20 classes

## __Parameter configuration__

#### Learner hyper-parameters:
- Outer learning rate
- Inner learning rate
- Number of adaptation steps
- Number of batches to adapt to during one iteration / epoch
- Number of iterations

#### Experiment parameters:
- Ways (=number of classes to train on per batch)
- Shots (=number of samples per class to learn from)


## __Experiment__
A wrapper class used for logging and saving results and models


1. Set seed to modules
2. Initialize GPU & CUDA if available
3. Fetch vision dataset in the form of task datasets. Same as a dataset but organized in “task batches”.

## __Model__
Based on the dataset create a suitable network.

## __l2l.algorithms.MAML__

A wrapper class for a network that augments it with a clone function and an adapt function.

#### _Clone_

Makes a clone of the network in order to populate the buffer of the original network. The computational graph is kept, and you can compute the derivatives of the new modules' parameters w.r.t the original parameters.


#### _Adapt_
Calculate and update the model’s parameters in place.

1. Gather only the differentiable parameters and set to None the gradients of the non differentiable parameters
2. Differentiate (calculate gradients of) the loss w.r.t to the parameters using autograd.grad()
    1. **Loss = outputs** (sequence of Tensor) – outputs of the differentiated function.
    2. **Diff params = inputs** (sequence of Tensor) – Inputs w.r.t. which the gradient will be returned (and not accumulated into .grad).

3. Perform a MAML update


#### _Update_
Performs a MAML update on model using grads and lr. The function re-routes the Python object, thus avoiding in-place operations. NOTE: The model itself is updated in-place (no deepcopy), but the parameters' tensors are not.


## __Optimizer__
As a default optimizer for the network we use Adam.

It is a first-order stochastic optimization method using gradients to update the weights of the network. We use two functions, zero_grad() which clears out the gradients and initializes them to zero and step()

#### _Step_
https://pytorch.org/docs/stable/optim.html


## __Loss function__
As a default classification loss function we use Cross Entropy with mean reduction.

Calculates the loss between predictions and actual labels.
Then calculates the gradients of the parameters using backward().
https://pytorch.org/docs/stable/nn.html#torch.nn.CrossEntropyLoss 

## __Outer loop__
At each iteration of the outer loop all of the parameters of the network are updated based on the accumulated gradients of the inner loop and using the Adam optimizer.

1. Accumulate gradients in MAML’s parameters / computation graph by adapting to multiple batches with the inner loop

2. Average out / Divide the gradients by the number of batches

3. Take an Optimizer.step()

## __Inner loop__

1. We make a clone of the model since we are planning on adapting (calculating gradients and updating the model based on them) to new data but we don’t want to actually update our base network.

2. Sample a random batch of images & labels as a list with two tensors, the first is the actual pixel values, the second is the labels

3. Adapt the model (“learner”) to the a batch of data and get the evaluation loss

4. Calculate the gradients of the updated parameters of the network on the unseen evaluation data.

5. Make another clone of the network

6. Sample another batch from the validation set

7. Adapt to the validation set to monitor the model’s progress

## __Fast adapt__

1. Split the batch of data to train data and evaluation data (aka **Query set!**).
This evaluation batch is usually called “query set” in meta learning papers.
This is done because in MAML we:

    1. Pass a training batch through the network
    2. Calculate the loss
    3. Calculate the gradients
    4. Update the parameters based on that (training) loss
    5. Pass an evaluation batch through the network
    6. Calculate the evaluation loss
    7. Calculate the new gradients from the unseen data which will be later accumulated to be used for the OUTER model update.

+(ANIL version) Pass the data through the network and get the features before the head as data input for the head to adapt to.

2. Adapt the network to the train data
    1. Pass the data to the network and get the predictions
    2. Calculate the loss (cross-entropy) of the predictions & actual labels
    3. Call MAML.adapt() to update the model’s parameters based on the loss

3. Make predictions for the validation data and calculate the validation loss & accuracy
