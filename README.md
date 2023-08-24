# DeepGCF

DeepGCF learns the functional conservation between pig and human based on the epigenome profiles from two species. It can be applied to pairwise comparisons of any two species given their epigenome profiles sequence alignments.

## Human-pig DeepGCF score

DeepGCF scores for human (GRCh38) and pig (susScr11) are available [here](https://farm.cse.ucdavis.edu/~liangend/DeepGCF/). In the human bed file, each window has a unique score, but in the pig bed file, there are overlapping windows (the same region may have multiple scores) because the input sequence alignment is generated using human genome as the basis, which could align to multiple segments of pig genome.

## Training DeepGCF model
To train DeepGCF, Python 3.6 or above is recommended, and python packages `sklearn`, `selene-sdk`, `torch`, and `numpy` are required. Scripts involving data preparation and training DeepGCF are modified from [LECIF](https://github.com/ernstlab/LECIF). 

[1. Training DeepSEA](https://github.com/liangend/DeepGCF/tree/main/tutorial/1_train_deepsea.md)

[2. Data preparation](https://github.com/liangend/DeepGCF/tree/main/tutorial/2_data_prep.md)

[3. Feature predition using DeepSEA](https://github.com/liangend/DeepGCF/tree/main/tutorial/3_pred_deepsea.md)

[4. Training DeepGCF](https://github.com/liangend/DeepGCF/tree/main/tutorial/4_train_deepgcf.md)

[5. Prediction using DeepGCF](https://github.com/liangend/DeepGCF/tree/main/tutorial/5_pred_deepgcf.md)

## Reference
Li et al., Learning functional conservation between human and pig to decipher evolutionary mechanisms underlying gene expression and complex traits, Cell Genomics (2023), https://doi.org/10.1016/j.xgen.2023.100390


