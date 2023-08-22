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

