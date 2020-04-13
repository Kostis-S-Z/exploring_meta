# Walkthrough of the code for MAML & ANIL in RL experiments


## __Definitions__
- Iterations = epochs
- Network = model ~ policy
- Inner lr (α): step value for inner loop adaptation of MAML / ANIL
- Outer lr (β): step value for outer loop optimization of the the model

## __Environments__

### Particles 2D

A point in a 2D bounded continuous space receives directional force and moves accordingly.

**Task**:  Reach the goal cooardinates. (Randomly generated)
 
**Reward**:  Negative distance from the goal.

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
