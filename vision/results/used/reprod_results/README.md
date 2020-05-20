## MAML & ANIL

### Mini ImageNet ( 5-way, 1-shot )

These models were trained with the "default" (same as reported in the paper) hyper-parameter settings for 5-way, 1-shot classification of Mini ImageNet.

For scale, a completely random network with only inner loop adaptation during testing, can achieve **~35%** accuracy.

#### Hyper-parameters
. | Seed | Outer LR | Inner LR | Inner steps | Inner steps eval| Iterations | Batch Size
--- | --- | --- | --- | --- | --- | --- | --- |
MAML | 42 | 0.003 | 0.01 | 5 | 10 | 60.000 | 4
ANIL | 42 | 0.001 | 0.01 | 5 | 10 | 60.000 | 4


#### Results
. | Train Acc | Valid Acc | Test Acc
--- | --- | --- | --- |
MAML Original | - | - | 48.7
MAML Learn2Learn | - | - | 48.3
This MAML Model | 70 | 46.3 | 50.6
ANIL Original | - | - | 46.3+0.4
This ANIL Model | 85 | 45 | 43.1

### Mini ImageNet ( 5-way, 5-shot )

#### Hyper-parameters
. | Seed | Outer LR | Inner LR | Inner steps | Inner steps eval| Iterations | Batch Size
--- | --- | --- | --- | --- | --- | --- | --- |
MAML | 42 | 0.003 | 0.01 | 5 | 10 | 60.000 | 2
ANIL | 42 | 0.001 | 0.01 | 5 | 10 | 60.000 | 2

### Omniglot ( 5-way, 1 or 5-shot )

#### Hyper-parameters
. | Seed | Outer LR | Inner LR | Inner steps | Inner steps eval| Iterations | Batch Size
--- | --- | --- | --- | --- | --- | --- | --- |
MAML | 42 | 0.003 | 0.4 | 1 | 3 | 60.000 | 32

### Mini ImageNet ( 20-way, 1 or 5-shot )

#### Hyper-parameters
. | Seed | Outer LR | Inner LR | Inner steps | Inner steps eval| Iterations | Batch Size
--- | --- | --- | --- | --- | --- | --- | --- |
MAML | 42 | 0.003 | 0.1 | 5 | 10 | 60.000 | 16

### RL: 2D Navigation (or Particles 2D)

. | Seed | Outer LR | Inner LR | Inner LR test | Inner steps | Inner steps eval| Iterations | Batch Size
--- | --- | --- | --- | --- | --- | --- | --- | --- |
MAML | 42 | 0.003 | 0.1 | 0.1 (1st update), 0.05 (after) | 1 | 1(?) | 500 | 20
