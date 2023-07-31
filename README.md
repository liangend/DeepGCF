# DeepGCF
DeepGCF learns the functional conservation between pig and human based on the epigenome profiles from two species. It can be applied to pairwise comparisons of any two species given their epigenome profiles sequence alignments.

## Human-pig DeepGCF score
DeepGCF scores for human (GRCh38) and pig (susScr11) are available [here](https://farm.cse.ucdavis.edu/~liangend/DeepGCF/). In the human bed file, each window has a unique score, but in the pig bed file, there are overlapping windows (the same region may have multiple scores) because the input sequence alignment is generated using human genome as the basis, which could align to multiple segments of pig genome.

## Training DeepGCF model
### 1. Training DeepSEA
The first step of training DeepGCF is to convert binary functional features to continuous values through a deep convolutional network [DeepSEA](https://www.nature.com/articles/nmeth.3547), which is implemented in a Python-based package [Selene](https://github.com/FunctionLab/selene). To run Selene,  Python 3.6 or above is recommended, and python packages `selene-sdk`, `torch`, and `numpy` are required.

DeepSEA predicts the binary functional features using genome sequences as the input. Briefly, DeepSEA requires 4 inputs: 

1) Reference genome (.fasta); 

2) Functional features (.bed.gz) with 4 columns: chromosome, start bp, end bp, name of the feature. Detailed instructions of preparing a functional feature file can be found [here](https://github.com/FunctionLab/selene/blob/master/tutorials/getting_started_with_selene/getting_started_with_selene.ipynb); 

3) Distinct feature (.txt) with one column of distinct feature names matching the feature names in the functional feature file; 

4) Training intervals (.txt). The interval samples for training. Intervals should contain at least 1 functional feature specified in the functional feature file.

Here is an example of training DeepSEA for one single functional feature. First download and extract the hg19 reference genome:
```
wget https://www.encodeproject.org/files/male.hg19/@@download/male.hg19.fasta.gz
gzip -d male.hg19.fasta.gz
```
The other 3 required input files can be found in [DeepSEA_example](https://github.com/liangend/DeepGCF/tree/main/DeepSEA_example). You also need a Python script `deeperdeepsea.py`, which constructs the DeepSEA structure. Then by specifying the input files and other model parameters in a configuration file (`simple_train.yml`), we can run DeepSEA as follows in Python:
```
from selene_sdk.utils import load_path
from selene_sdk.utils import parse_configs_and_run
configs = load_path("./simple_train.yml")
parse_configs_and_run(configs, lr=0.01)
```
`lr` specifies the learning rate to be 0.01. In the output directory, you can find the model trained by DeepSEA (.tar) and model performance evaluated on the test set. Detailed tutorial of DeepSEA can be found [here](https://github.com/FunctionLab/selene). 
