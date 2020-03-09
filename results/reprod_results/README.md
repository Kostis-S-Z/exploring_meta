## MAML

### Mini ImageNet

ID: maml_min_04_03_16h43_42_2373

This model was trained with the "default" (same as reported in the paper) hyper-parameter settings for 5-way, 1-shot classification of Mini ImageNet.

For scale, a completely random network with only inner loop adaptation during testing, can achieve **~35%** accuracy.

#### Hyper-parameters
Seed | Inner LR | Outer LR | Inner steps | Iterations | Batch Size
--- | --- | --- | --- | --- | --- 
42 | 0.003 | 0.5 | 1 | 30.000 | 32

#### Results
. | Train Acc | Valid Acc | Test Acc
--- | --- | --- | --- |
Original | - | - | 48.7
Learn2Learn | - | - | 48.3
This Model | 70 | 46.3 | 50.6

#### CL Matrix

. | Te1 | Te2 | Te3 | Te4 | Te5
--- | --- | --- | --- | --- | --- |
Tr1 | 1. | 0.4 | 0. | 0.2 | 0.4
Tr2 | 0.6 |  1. | 0.1 | 0. | 0.3
Tr3 | 0.1 | 0.1 | 1. | 0.6 | 0.1
Tr4 | 0.1 | 0. | 0.5 | 1. | 0.
Tr5 | 0.5 | 0.3 | 0.1 | 0. | 1.