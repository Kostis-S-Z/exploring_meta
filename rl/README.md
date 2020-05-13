# Walkthrough of the code for MAML & ANIL in RL experiments


## __Definitions__
- Iterations = epochs
- Network = model ~ policy
- Inner lr (α): step value for inner loop adaptation of MAML / ANIL
- Outer lr (β): step value for outer loop optimization of the the model

- Adapt steps: how many times you will replay & learn a specific number of episodes (=adapt_batch_size)

- Meta_batch_size (=ways): how many tasks an epoch has. (a task can have one or many episodes)

- Adapt_batch_size (=shots): number of episodes (not steps!) during adaptation

## __Environments__

### Particles 2D

A point in a 2D bounded continuous space receives directional force and moves accordingly.

**Task**:  Reach the goal cooardinates. (Randomly generated)
 
**Reward**:  Negative distance from the goal.


### MetaWorld

#### Default parameters of MAML-TRPO for ML1, ML10:

```
inner_lr = 0.1
adapt_steps = 1
tau / gae_lambda = 1.0
gamma / discount = 0.99
adapt_batch_size = 10
meta_batch_size = 20
iterations = 300
``` 

### [Procgen](https://openai.com/blog/procgen-benchmark/)

+++

#### Coinrun

+++

## __Parameter configuration__

#### Learner hyper-parameters:
- Outer learning rate
- Inner learning rate


## __Experiment__
A wrapper class used for logging and saving results and models


1. Set seed to modules
2. Initialize GPU & CUDA if available
3. Fetch vision dataset in the form of task datasets. Same as a dataset but organized in “task batches”.
