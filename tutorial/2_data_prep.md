### 2. Data preparation

There are several files required before DeepGCF training.

#### Genome alignment

This step generates the alignment of 50-bp window between human and pig.

1.  Download [human-pig genome alignment (axtNet file)](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/vsSusScr11/) with human as the reference. Alignment between other species can also be found on [UCSC genome browser](https://hgdownload.soe.ucsc.edu/downloads.html) or be made by [lastz](https://github.com/lastz/lastz).

2.  Find all pig sequences that align to human chromosome. This step requires axtNet file, [human](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes) and [pig chromosome sizes](https://hgdownload.soe.ucsc.edu/goldenPath/susScr11/bigZips/susScr11.chrom.sizes) as inputs.

```         
python src/findAligningBases.py [-h] -a AXTNET_FILE -hg HUMAN_CHROM_SIZE_FILE -m PIG_CHROM_SIZE_FILE -o OUTPUT_FILENAME

optional arguments:
   -h, --help            show this help message and exit

required arguments:
   -a, --axtnet-filename: path to axtNet file name
   -hg, --human-chrom-size-filename: path to human chromosome size file name
   -m, --pig-chrom-size-filename: path to chromosome size file name
   -o, --output-filename: path to output file name

# Example: python src/findAligningBases.py -a ~/hg38.susScr11.net.axt.gz -hg ~/hg38.chrom.sizes -m ~/susScr11.chrom.sizes -o aligning_bases/hg38.susScr11.alignbase.gz
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
 
# Example: python src/samplePairs.py -i position/hg38.susScr11.basepair.gz -o position/hg38.susScr11.50bp
```

Three output files are generated: (i) a file with human and pig regions that align to each other (`hg38.susScr11.50bp.gz`), (ii) a file with human regions only (`hg38.susScr11.50bp.h.gz`), and (iii) a file with pig regions only (`hg38.susScr11.50bp.m.gz`).

#### Training, validation, testing sets split
We split the whole genome into training (for model training), validation (for model evaluation during training), and testing (generate ROC and PRC after training) sets for odd and even (including X chromosome) chromosomes, respectively. We conduct the data split to make sure there is no data leakage (no overlap among training, testing and validation).

1. Data split
```
python src/splitData.py [-h] -A HUMAN_REGION_FILENAME -B PIG_REGION_FILENAME 
                             -N TOTAL_NUM_PAIRS -o OUTPUT_DIR
                             --odd-training-human-chrom ODD_TRAINING_HUMAN_CHROM
                             --odd-validation-human-chrom ODD_VALIDATION_HUMAN_CHROM
                             --odd-test-human-chrom ODD_TEST_HUMAN_CHROM
                             --odd-prediction-human-chrom ODD_PREDICTION_HUMAN_CHROM
                             --odd-training-pig-chrom ODD_TRAINING_PIG_CHROM
                             --odd-validation-pig-chrom ODD_VALIDATION_PIG_CHROM
                             --odd-test-pig-chrom ODD_TEST_PIG_CHROM
                             --even-training-human-chrom EVEN_TRAINING_HUMAN_CHROM
                             --even-validation-human-chrom EVEN_VALIDATION_HUMAN_CHROM
                             --even-test-human-chrom EVEN_TEST_HUMAN_CHROM
                             --even-prediction-human-chrom EVEN_PREDICTION_HUMAN_CHROM
                             --even-training-pig-chrom EVEN_TRAINING_PIG_CHROM
                             --even-validation-pig-chrom EVEN_VALIDATION_PIG_CHROM
                             --even-test-pig-chrom EVEN_TEST_PIG_CHROM

optional arguments:
   -h, --help: show this help message and exit
 
required arguments specifying input and output:
   -A, --human-region-filename: path to human region filename
   -B, --pig-region-filename: path to pig region filename
   -N, --total-num-pairs: total number of pairs (number of lines of input files)
   -o, --output-dir: output directory
 
 required arguments specifying the split:
   --odd-training-human-chrom: human chromosomes to include in odd training data
   --odd-validation-human-chrom: human chromosomes to include in odd validation data
   --odd-test-human-chrom: human chromosomes to include in odd test data
   --odd-prediction-human-chrom: human chromosomes to include in odd prediction data
   --odd-training-pig-chrom: pig chromosomes to include in odd training data
   --odd-validation-pig-chrom: pig chromosomes to include in odd validation data
   --odd-test-pig-chrom: pig chromosomes to include in odd test data
   --even-training-human-chrom: human chromosomes to include in even training data
   --even-validation-human-chrom: human chromosomes to include in even validation data
   --even-test-human-chrom: human chromosomes to include in even test data
   --even-prediction-human-chrom: human chromosomes to include in even prediction data
   --even-training-pig-chrom: pig chromosomes to include in even training data
   --even-validation-pig-chrom: pig chromosomes to include in even validation data
   --even-test-pig-chrom: pig chromosomes to include in even test data
 
# Example: python src/splitData.py -A position/hg38.susScr11.50bp.h.gz \
            -B position/hg38.susScr11.50bp.m.gz \
            -N 40268928 -o data/ \
            --odd-training-human-chrom chr1 chr5 chr7 chr9 chr13 chr17 chr21 \
       	    --odd-validation-human-chrom chr3 chr11 chr15 chr19 \
       	    --odd-test-human-chrom chr1 chr3 chr5 chr7 chr9 chr11 chr13 chr15 chr17 chr19 chr21 \
       	    --odd-prediction-human-chrom chr1 chr3 chr5 chr7 chr9 chr11 chr13 chr15 chr17 chr19 chr21 \
       	    --odd-training-pig-chrom chr1 chr5 chr7 chr9 chr13 chr17 \
       	    --odd-validation-pig-chrom chr3 chr11 chr15 \
       	    --odd-test-pig-chrom chr1 chr3 chr5 chr7 chr9 chr11 chr13 chr15 chr17 \
       	    --even-training-human-chrom chr2 chr6 chr10 chr14 chr18 chr22 \
       	    --even-validation-human-chrom chr4 chr8 chr12 chr16 chr20 \
       	    --even-test-human-chrom chr2 chr4 chr6 chr8 chr10 chr12 chr14 chr16 chr18 chr20 chr22 \
       	    --even-prediction-human-chrom chr2 chr4 chr6 chr8 chr10 chr12 chr14 chr16 chr18 chr20 chr22 chrX \
       	    --even-training-pig-chrom chr2 chr6 chr10 chr14 chr18 \
	          --even-validation-pig-chrom chr4 chr8 chr12 chr16 \
	          --even-test-pig-chrom chr2 chr4 chr6 chr8 chr10 chr12 chr14 chr16 chr18
```
There are 16 output files, including 12 files of training, validation, and testing sets of human and pig regions from odd and even chromosomes, 2 files (`odd_all.h.gz` and `odd_all.m.gz`) for the prediction of human regions from odd chromosomes and corresponding paired pig regions using the model trained from even chromosomes, and 2 files (`even_all.h.gz` and `even_all.m.gz`) for the prediction of human regions from odd chromosomes and corresponding paired pig regions using the model trained from even chromosomes.

2. Next we need to shuffle the training, validation, and testing sets to randomize the order of aligning pairs (while keeping the human-pig pairs intact), and shuffle them again but only with the pig regions to generate non-orthologous pairs.
```
src/prepareData \
   <path to directory with output files from splitData.py> \
   <path to output directory to store shuffled/sampled data files>			
   
# Example: src/prepareData data/ data/
```
There are 18 output region files, including 12 files (start with `shuf_`) of shuffled training, validation, and testing sets for human and pig, and 6 files (start with `shuf2x_`) of pig files shuffled twice.




