## MAML & ΑΝΙΛ

### Mini ImageNet ( 5-way, 1-shot )

ID: maml_min_04_03_16h43_42_2373

This model was trained with the "default" (same as reported in the paper) hyper-parameter settings for 5-way, 1-shot classification of Mini ImageNet.

For scale, a completely random network with only inner loop adaptation during testing, can achieve **~35%** accuracy.

#### Hyper-parameters
. | Seed | Inner LR | Outer LR | Inner steps | Iterations | Batch Size
--- | --- | --- | --- | --- | --- | --- 
MAML | 42 | 0.003 | 0.5 | 1 | 30.000 | 32
ANIL | 42 | 0.001 | 0.1 | 1 | 30.000 | 32


#### Results
. | Train Acc | Valid Acc | Test Acc
--- | --- | --- | --- |
MAML Original | - | - | 48.7
MAML Learn2Learn | - | - | 48.3
This MAML Model | 70 | 46.3 | 50.6
ANIL Original | - | - | 46.3+0.4
This ANIL Model | 85 | 45 | 43.1