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



