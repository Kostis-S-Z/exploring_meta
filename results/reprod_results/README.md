## MAML

### Mini ImageNet

ID: maml_min_04_03_16h43_42_2373

This model was trained with the "default" (same as reported in the paper) hyper-parameter settings for 5-way, 1-shot classification of Mini ImageNet.

#### Hyper-parameters
Seed | Inner LR | Outer LR | Adaptation steps | Iterations | Batch Size
--- | --- | --- | --- | --- | --- 
42 | 0.003 | 0.5 | 1 | 30.000 | 32

#### Results
. | Train Acc | Valid Acc | Test Acc
--- | --- | --- | --- |
Original | - | - | 48.7
Learn2Learn | - | - | 48.3
This Model | 70 | 46.3 | 50.6