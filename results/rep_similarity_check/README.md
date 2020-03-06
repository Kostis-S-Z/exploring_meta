Compare the representation of the model before any training (random weights) and after a few training steps.

1. A comparison between the representation before and after the whole training step. A train iteration consists of introducing a new batch (n-ways, k-shots) with inner + outer loop adaptation. 

![alt text](https://github.com/Kostis-S-Z/exploring_meta/blob/master/results/rep_similarity_check/CCA_evolution_outer.png "CCA Similarity")
![alt text](https://github.com/Kostis-S-Z/exploring_meta/blob/master/results/rep_similarity_check/Linear_CKA_evolution_outer.png "Linear CKA Similarity")
![alt text](https://github.com/Kostis-S-Z/exploring_meta/blob/master/results/rep_similarity_check/Kernel_CKA_evolution_outer.png "Kernel CKA Similarity")

_Across representations in the outer loop it is expected that the representation is not changing as fast due to the much lower learning rate._

- Inner loop learning rate: 0.5
- Outer loop learning rate: 0.003

2. A comparison between the representation before and after just the **inner** loop adaptation. One inner loop step consists of X number of adaptation steps in the same batch

    1. Randomly initialized weights: Before and After inner loop adaptation at a random network. We expect the similarity to be really low since the parameter space has not been tuned for adaptation. 

    ![alt text](https://github.com/Kostis-S-Z/exploring_meta/blob/master/results/rep_similarity_check/inner_CCA_evolution_random.png "CCA Similarity")
    ![alt text](https://github.com/Kostis-S-Z/exploring_meta/blob/master/results/rep_similarity_check/inner_Linear_CKA_evolution_random.png "Linear CKA Similarity")
    ![alt text](https://github.com/Kostis-S-Z/exploring_meta/blob/master/results/rep_similarity_check/inner_Kernel_CKA_evolution_random.png "Kernel CKA Similarity")
    
    2. After a few meta train iterations: Even after 50 train iterations the representation is much more robust than random. This is because even after 10 inner loop adaptation steps with 0.5 learning rate, the similarity measure is higher compared to the random.

    ![alt text](https://github.com/Kostis-S-Z/exploring_meta/blob/master/results/rep_similarity_check/inner_CCA_evolution_pretrained.png "CCA Similarity")
    ![alt text](https://github.com/Kostis-S-Z/exploring_meta/blob/master/results/rep_similarity_check/inner_Linear_CKA_evolution_pretrained.png "Linear CKA Similarity")
    ![alt text](https://github.com/Kostis-S-Z/exploring_meta/blob/master/results/rep_similarity_check/inner_Kernel_CKA_evolution_pretrained.png "Kernel CKA Similarity")


_The steep decrease of similarity indicates that the representation does indeed change substantially even in such a few adaptation steps._


FILE INDEX:

- CCA_evolution_outer -> maml_min_06_03_11h58_42_7964
- Kernel_CKA_evolution_outer -> maml_min_06_03_11h58_42_7964
- Linear_CKA_evolution_outer -> maml_min_06_03_11h58_42_7964


- inner_CCA_evolution_random -> maml_min_06_03_12h32_42_5841
- inner_Kernel_CKA_evolution_random -> maml_min_06_03_12h32_42_5841
- inner_Linear_CKA_evolution_random -> maml_min_06_03_12h32_42_5841


- inner_CCA_evolution_pretrained -> maml_min_06_03_12h37_42_1182
- inner_Kernel_CKA_evolution_pretrained -> maml_min_06_03_12h37_42_1182
- inner_Linear_CKA_evolution_pretrained -> maml_min_06_03_12h37_42_1182
