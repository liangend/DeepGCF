# DeepGCF

DeepGCF learns the functional conservation between pig and human based on the epigenome profiles from two species. It can be applied to pairwise comparisons of any two species given their epigenome profiles sequence alignments.

## Human-pig DeepGCF score

DeepGCF scores for human (GRCh38) and pig (susScr11) are available [here](https://farm.cse.ucdavis.edu/~liangend/DeepGCF/). In the human bed file, each window has a unique score, but in the pig bed file, there are overlapping windows (the same region may have multiple scores) because the input sequence alignment is generated using human genome as the basis, which could align to multiple segments of pig genome.

## Training DeepGCF model

### 1. Training DeepSEA

The first step of training DeepGCF is to convert binary functional features to continuous values through a deep convolutional network [DeepSEA](https://www.nature.com/articles/nmeth.3547), which is implemented in a Python-based package [Selene](https://github.com/FunctionLab/selene). To run Selene, Python 3.6 or above is recommended, and python packages `selene-sdk`, `torch`, and `numpy` are required.

DeepSEA predicts the binary functional features using genome sequences as the input. Briefly, DeepSEA requires 4 inputs:

1)  Reference genome (.fasta);

2)  Functional features (.bed.gz) with 4 columns: chromosome, start bp, end bp, name of the feature. Detailed instructions of preparing a functional feature file can be found [here](https://github.com/FunctionLab/selene/blob/master/tutorials/getting_started_with_selene/getting_started_with_selene.ipynb);

3)  Distinct feature (.txt) with one column of distinct feature names matching the feature names in the functional feature file;

4)  Training intervals (.txt). The interval samples for training. Intervals should contain at least 1 functional feature specified in the functional feature file.

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
   -m, --pig-chrom-size-filename: path to chromosome size file name
   -o, --output-filename: path to output file name

# Example: python src/findAligningBases.py -a ~/hg38.susScr11.net.axt.gz -m ~/susScr11.chrom.sizes -o aligning_bases/hg38.susScr11.alignbase.gz
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

### 3. Feature predition using DeepSEA

### 4. Training DeepGCF

Finally, we can train the DeepGCF model using the functional features predicted using DeepSEA as the inputs. To make predictions for human regions from even chromosomes or the X chromosome and the corresponding paired pig regions, we train a DeepGCF model using paired regions from odd human and pig chromosomes. Similarly, paired regions from even human and pig chromosomes are used to train the model for predictions on human regions from odd chromosomes and the corresponding paired pig regions. Example input files for training DeepGCF using odd chromosomes are provided [here](https://github.com/liangend/DeepGCF/tree/main/DeepGCF_example).

1.  Hyper-parameter search. Train N neural networks, each with randomly selected combinations of hyper-parameters and trained on the same subset of the whole training set.
```         
python src/train_deepgcf.py [-h] [-o OUTPUT_FILENAME_PREFIX] [-k] [-v] [-t]
                  [-r NEG_DATA_RATIO] [-s SEED] [-e NUM_EPOCH] 
                  -A HUMAN_TRAINING_DATA_FILENAME -B PIG_TRAINING_DATA_FILENAME 
                  -C SHUFFLED_PIG_TRAINING_DATA_FILENAME -D HUMAN_VALIDATION_DATA_FILENAME 
                  -E PIG_VALIDATION_DATA_FILENAME -F SHUFFLED_PIG_VALIDATION_DATA_FILENAME
                  [-tr POSITIVE_TRAINING_DATA_SIZE]
                  [-tra TOTAL_POSITIVE_TRAINING_DATA_SIZE]
                  [-va POSITIVE_VALIDATION_DATA_SIZE]
                  [-hf NUM_HUMAN_FEATURES]
                  [-mf NUM_PIG_FEATURES]
                  [-hrmin HUMAN_RNASEQ_MIN]
                  [-hrmax HUMAN_RNASEQ_MAX]
                  [-mrmin PIG_RNASEQ_MIN]
                  [-mrmax PIG_RNASEQ_MAX] [-b BATCH_SIZE]
                  [-l LEARNING_RATE] [-d DROPOUT_RATE]
                  [-nl1 NUM_LAYERS_1] [-nl2 NUM_LAYERS_2]
                  [-nnh1 NUM_NEURON_HUMAN_1]
                  [-nnh2 NUM_NEURON_HUMAN_2]
                  [-nnm1 NUM_NEURON_PIG_1]
                  [-nnm2 NUM_NEURON_PIG_2] [-nn1 NUM_NEURON_1]
                  [-nn2 NUM_NEURON_2]
 
 optional arguments:
   -h, --help: show this help message and exit
   -o, --output-filename-prefix: output prefix (must be specified if saving (-v))
   -k, --random-search: if hyper-parameters should be randomly set
   -v, --save: if the trained classifier should be saved after training
   -t, --early-stopping: if early stopping should be allowed (stopping before the maximum number of epochs if there is no improvement in validation AUROC in three consecutive epochs)
   -r, --neg-data-ratio: weight ratio of non-orthologous samples to orthologous samples (default: 50)
   -s, --seed: random seed (default: 1)
   -e, --num-epoch: maximum number of training epochs (default: 100)
 
 required arguments specifying training data:
   -A, --human-training-data-filename: path to human training data file
   -B, --pig-training-data-filename: path to pig orthologous training data file
   -C, --shuffled-pig-training-data-filename: path to pig shuffled/non-orthologous training data file
 
 required arguments specifying validation data:
   -D, --human-validation-data-filename: path to human validation data file
   -E, --pig-validation-data-filename: path to pig orthologous validation data file
   -F, --shuffled-pig-validation-data-filename: path to pig shuffled/non-orthologous validation data file
 
 required arguments describing feature data:
   -tra, --total-positive-training-data-size: number of samples in total orthologous training data to *read*
   -tr, --positive-training-data-size: number of samples in orthologous training data to *use* (default: 1000000)
   -va, --positive-validation-data-size: number of samples in orthologous validation data to use (default: 100000)
   -hf, --num-human-features: number of human features in input vector
   -mf, --num-pig-features: number of pig features in input vector
   -hrmin, --human-rnaseq-min: minimum expression level in human RNA-seq data
   -hrmax, --human-rnaseq-max: maximum expression level in human RNA-seq data
   -mrmin, --pig-rnaseq-min: minimum expression level in pig RNA-seq data
   -mrmax, --pig-rnaseq-max: maximum expression level in pig RNA-seq data
 
 optional arguments specifying hyper-parameters (ignored if random search (-k) is specified):
   -b, --batch-size: batch size (default: 128)
   -l, --learning-rate: learning rate (default: 0.1)
   -d, --dropout-rate: dropout rate (default: 0.1)
   -nl1, --num-layers-1: number of hidden layers in species-specific sub-networks (default: 1)
   -nl2, --num-layers-2: number of hidden layers in final sub-network (default: 1)
   -nnh1, --num-neuron-human-1: number of neurons in the first hidden layer in the human-specific sub-network (default :1)
   -nnh2, --num-neuron-human-2: number of neurons in the second hidden layer in the human-specific sub-network (default: 0)
   -nnm1, --num-neuron-pig-1: number of neurons in the first hidden layer in the pig-specific sub-network (default: 128)
   -nnm2, --num-neuron-pig-2: number of neurons in the second hidden layer in the pig-specific sub-network (default: 0)
   -nn1, --num-neuron-1: number of neurons in the first hidden layer in the final sub-network (default: 256)
   -nn2, --num-neuron-2: number of neurons in the second hidden layer in the final sub-network (default: 0)
   
# Example: python src/train_deepgcf.py \
      -A DeepGCF_example/shuf_odd_training_example.h.tsv \
            -B DeepGCF_example/shuf_odd_training_example.m.tsv \
            -C DeepGCF_example/shufx2_odd_training_example.m.tsv \
            -D DeepGCF_example/shuf_odd_validation_example.h.tsv \
            -E DeepGCF_example/shuf_odd_validation_example.m.tsv \
            -F DeepGCF_example/shufx2_odd_validation_example.m.tsv \
            -tra 5000 -tr 5000 -va 4999 \
            -hf 14 -mf 10 \
            -s 1 -t -k > output/NN_search_odd1.txt
```

`-k` specifies the hyper-parameter search based on a random grid search. [This table](https://github.com/liangend/DeepGCF/tree/main/hyper_params.xlsx) lists all the candidate hyper-parameters. Both orthologous training data size (specified by -tr) and total orthologous training data size (specified by -tra) is set to 5000 since we want all neural networks to be trained on the first 5000 orthologous samples and the first 5000 non-orthologous samples in the provided training data file. It is assumed that the number of non-orthologous samples is the same as the number of orthologous samples. The last line of the output file `NN_search_odd1.txt` contains 32 columns as follows:

-   Columns 1-6: seed, non-orthologous to orthologous sample weight ratio, number of orthologous training samples, number of non-orthologous training samples, number of orthologous validation samples, number of non-orthologous validation samples

-   Columns 7-9: batch size, learning rate (epsilon), dropout rate

-   Columns 10-12: number of hidden layers in the human-specific sub-network, pig-specific sub-network, and final sub-network

-   Columns 13,14: number of neurons in the first and second hidden layers in the human-specific sub-network

-   Columns 15,16: number of neurons in the first and second hidden layers in the pig-specific sub-network

-   Columns 17,18: number of neurons in the first and second hidden layers in the final sub-network

-   Columns 19,20: current epoch and training loss

-   Columns 21-26: training MSE, training AUROC, training AUPRC, mean prediction for all training samples, mean prediction for orthologous training samples, mean prediction for non-orthologous training samples

-   Columns 27-32: validation MSE, validation AUROC, validation AUPRC, mean prediction for all validation samples, mean prediction for orthologous validation samples, mean prediciton for non-orthologous validation samples

Repeat this step by N times (e.g., 100), each time with a different seed number (`-s`), then find the best combination of hyper-parameters based on the validation AUROC.

2. Train the DeepGCF model using all the training samples based on the best combination of hyper-parameters. An example is as follows:
```
# Example of training a DeepGCF model:
python src/train_deepgcf.py \
      -A DeepGCF_example/shuf_odd_training_example.h.tsv \
      -B DeepGCF_example/shuf_odd_training_example.m.tsv \
      -C DeepGCF_example/shufx2_odd_training_example.m.tsv \
      -D DeepGCF_example/shuf_odd_validation_example.h.tsv \
      -E DeepGCF_example/shuf_odd_validation_example.m.tsv \
      -F DeepGCF_example/shufx2_odd_validation_example.m.tsv \
 			-o output/NN_odd \
 			-tra 9999 -tr 9999 -va 4999 \
 			-hf 14 -mf 10 \
 			-b 128 -l 0.01 -d 0 \
 			-nl2 1 -nn1 8 -nn2 0 \
 			-nnh1 128 -nnh2 0 -nnm1 128 -nnm2 0 \
 			-s 1 -t -v > output/NN_odd.log
```
In the output folder, there will be a trained model (`NN_odd_1_50_9999_9999_4999_4999_128_0.01_0.0_1_1_1_128_0_128_0_8_0.pt`). Repeat the hyper-parameter search step and model training step to train a model using paired functional features from even and X chromosomes.

### 5. Prediction using DeepGCF
As described in DeepGCF training section, model training and prediction are done twice for even and odd chromosomes separately. An example to predict the functional conservation for even chromosomes is as follows:
```
python src/predict_deepgcf.py [-h] [-b BATCH_SIZE] -t
                                 TRAINED_CLASSIFIER_FILENAME -H
                                 HUMAN_FEATURE_FILENAME -M
                                 PIG_FEATURE_FILENAME -d DATA_SIZE -o
                                 OUTPUT_FILENAME [-hf NUM_HUMAN_FEATURES]
                                 [-mf NUM_PIG_FEATURES]
                                 [-hrmin HUMAN_RNASEQ_MIN]
                                 [-hrmax HUMAN_RNASEQ_MAX]
                                 [-mrmin PIG_RNASEQ_MIN]
                                 [-mrmax PIG_RNASEQ_MAX]
 
 Generate predictions given a trained neural network
 
 optional arguments:
   -h, --help: show this help message and exit
   -b, --batch-size: batch size (default: 128)
 
 required arguments specifying input and output:
   -t, --trained-classifier-filename: path to a trained classifier (.pt)
   -H, --human-feature-filename: path to human feature data file
   -M, --pig-feature-filename: path to pig feature data file
   -d, --data-size: number of samples
   -o, --output-filename: path to output file
   -hf, --num-human-features: number of human features in input
   -mf, --num-pig-features: number of pig features in input
   -hrmin, --human-rnaseq-min: minimum expression level in human RNA-seq data
   -hrmax, --human-rnaseq-max: maximum expression level in human RNA-seq data
   -mrmin, --pig-rnaseq-min: minimum expression level in pig RNA-seq data
   -mrmax, --pig-rnaseq-max: maximum expression level in pig RNA-seq data
 
# Example: python src/predict_deepgcf.py \
  -t output/NN_odd_1_50_9999_9999_4999_4999_128_0.01_0.0_1_1_1_128_0_128_0_8_0.pt \
  -H DeepGCF_example/even_all_example.h.tsv \
  -M DeepGCF_example/even_all_example.m.tsv \
  -hf 14 -mf 10 \
  -d 100 -o output/even_all.gz
```
