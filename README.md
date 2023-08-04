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

### 2. Data preparation

There are several files required before DeepGCF training.

#### Genome alignment

This step generates the alignment of 50-bp window between human and pig. The code is based on [LECIF](https://github.com/ernstlab/LECIF) with small modifications.

1.  Download [human-pig genome alignment (axtNet file)](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/vsSusScr11/) with human as the reference. Alignment between other species can also be found on [UCSC genome browser](https://hgdownload.soe.ucsc.edu/downloads.html) or be made by [lastz](https://github.com/lastz/lastz).

2.  Find all pig sequences that align to human chromosome. This step requires axtNet file and [pig chromosome sizes](https://hgdownload.soe.ucsc.edu/goldenPath/susScr11/bigZips/susScr11.chrom.sizes) as inputs.

```         
python src/findAligningBases.py [-h] -a AXTNET_FILE -m CHROM_SIZE_FILE -o OUTPUT_FILENAME

optional arguments:
   -h, --help            show this help message and exit

required arguments:
   -a, --axtnet-filename: path to axtNet file name
   -m, --mouse-chrom-size-filename: path to chromosome size file name
   -o, --output-filename: path to output file name

# Example: python src/findAligningBases.py -a ~/hg38.susScr11.net.axt.gz -m ~/mm10.chrom.sizes -o aligning_bases/hg38.susScr11.alignbase.gz
```

3.  Aggregate all aligning pairs and assign a unique index to each.

```         
src/aggregateAligningBases \
   <directory of the output file from findAligningBases> \
   <path to output file name> 
 
# Example:
src/aggregateAligningBases aligning_bases/ position/hg38.susScr11.basepair.gz
```

4.  Sample the first base of every non-overlapping genomic window of length 50 bp (at most) defined across consecutive bases in each human chromosome that align to pig.

```         
python src/samplePairs.py [-h] [-b] -i INPUT_FILENAME -o OUTPUT_PREFIX
 
optional arguments:
   -h, --help: show this help message and exit
   -b, --bin-size: size (bp) of the non-overlapping genomic window (default: 50)
 
required arguments:
   -i, --input-filename: path to output filename from aggregateAligningBases containing aligning pairs of human and pig bases
   -o, --output-prefix: prefix for output files
 
# Example: python src/samplePairs.py -i position/hg19.susScr11.basepair.gz -o position/hg38.susScr11.50bp
```
