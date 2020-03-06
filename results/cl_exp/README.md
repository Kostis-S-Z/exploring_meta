_Continual Learning metrics as presented in: "Don’t forget, there is more than forgetting: new metrics for Continual Learning"_
paper link: _https://hal.archives-ouvertes.fr/hal-01951488/document_

Calculate metrics based on an accuracy matrix of N train tasks on N test tasks
The lower triangular matrix is the BWT, the higher triangular matrix is the FWT

The metrics below are taking into account the accuracy of the model at every timestep / task
by dividing the accuracies with the term "div" / "f_div" / "d_div"

  - Average Accuracy:  Accuracies across tesk tasks when trained on the last sample of train task i

  - Forward Transfer: Measures the influence that "learning" (training) a task i has on the performance (testing)
                      on a future task j.

  - Backward Transfer: Measures the influence that "learning" (training) a task i has on the performance (testing)
                       on a previous task j. This metric is split into two in order to better depict the concept
                       of catastrophic forgetting (or inversely, remembering) and positive backward transfer. backward transfer.
    - Remembering [-, 1]: 1 -> Perfect remembering / No (catastrophic) forgetting

    - BWT_plus [0, +]: 0 -> No improvement in previous tasks by training on new tasks
                       a positive value -> the higher the better. By itself it doesn't mean a lot, it is mostly used
                       for comparison between algorithms

Example:

For Tr1 = Te1 (test task 1 is the same samples and class as train task 1)

Accuracies matrix

. | Te1 | Te2 | Te3 | Te4 | Te5
--- | --- | --- | --- | --- | ---
**Tr1** | 1. | 0.1 | 0.1 | 0.1 | 0.2
**Tr2** | 0. | 1. | 0.2 | 0.1 | 0.3
**Tr3** | 0. | 0.1 | 1. | 0.2 |  0.3
**Tr4** | 0.4 | 0. | 0. | 1. | 0.2 
**Tr5** | 0.1 | 0.2 | 0.1 | 0.3 | 1. 

Average task accuracy | Forward Transfer | Remembering | Positive Backward Transfer
--- | --- | --- | ---
0.413 | 0.0 | 0.18 | -0.129
