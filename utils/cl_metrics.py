"""
Continual Learning metrics as presented in:
"Don’t forget, there is more than forgetting: new metrics for Continual Learning"

paper link: https://hal.archives-ouvertes.fr/hal-01951488/document
"""

import numpy as np


def cl_metrics(acc_matrix):
    """
    Calculate metrics based on an accuracy matrix of N train tasks on N test tasks
    The lower triangular matrix is the BWT, the higher triangular matrix is the FWT

    The metrics below are taking into account the accuracy of the model at every timestep / task
    by dividing the accuracies with the term "div" / "f_div" / "d_div"

      - Average Accuracy:  Accuracies across tesk tasks when trained on the last sample of train task i

      - Forward Transfer: Measures the influence that "learning" (training) a task i has on the performance (testing)
                          on a future task j.

      - Backward Transfer: Measures the influence that "learning" (training) a task i has on the performance (testing)
                           on a previous task j. This metric is split into two in order to better depict the concept
                           of catastrophic forgetting (or inversely, remembering) and positive backward transfer.
        - Remembering [0, 1]: 1 -> Perfect remembering / No (catastrophic) forgetting
                              0 -> Complete catastrophic forgetting

    :param acc_matrix: NxN matrix containing the model accuracy on a task j (y axis) after trained on task i (x axis)
    :return:dictionary of the results
    """
    n = acc_matrix.shape[0]

    # Average accuracy = Diagonal + Lower triangular
    av_acc_sum = np.tril(acc_matrix, k=0).sum()  # k=0 means include the diagonal
    div = (n * (n + 1)) / 2
    av_acc = av_acc_sum / div

    # Forward Transfer = Higher triangular
    f_acc_sum = np.triu(acc_matrix, k=1).sum()  # k=1 means do NOT include diagonal
    f_div = (n * (n - 1)) / 2
    fwt = f_acc_sum / f_div

    # Backward Transfer
    b_div = f_div
    b_acc_sum = 0
    for i in range(1, n):
        for j in range(n - 1):
            b_acc_sum += acc_matrix[i, j] - acc_matrix[j, j]

    bwt = b_acc_sum / b_div

    # Remembering: The higher the better (inverse of catastrophic forgetting / negative bwt)
    rem = 1 - np.abs(np.amin([bwt, 0]))

    # Positive backward transfer: The higher the better
    bwt_plus = np.amax([bwt, 0])

    return dict(av_acc=av_acc, fwt=fwt, rem=rem, bwt_plus=bwt_plus)
