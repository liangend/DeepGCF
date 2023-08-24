### 3. Feature predition using DeepSEA
We then need to predict the functional features for all the human and pig region files using DeepSEA. The following Python code can be used to generate the predictions.
```
# Example:

import selene_sdk
from selene_sdk.utils import DeeperDeepSEA
from selene_sdk.utils import NonStrandSpecific
from selene_sdk.predict import AnalyzeSequences
from selene_sdk.utils import load_features_list

model_architecture = NonStrandSpecific(DeeperDeepSEA(1000, 309))  ## 1000 is sequence length, which is specified in the configuration when training DeepSEA; 309 is the total number of distinct features
features = load_features_list("./distinct_features.txt")  # path to the distinct feature file
analysis = AnalyzeSequences(
    model_architecture,
    "deepsea_outputs/best_model.pth.tar", # path to the trained DeepSEA model
    sequence_length=1000,
    features=features,
    reference_sequence=selene_sdk.sequences.Genome("./GRCh38.fa"), # path to the reference genome
    use_cuda=False) # if GPU is available, changing use_cuda to True can increase the computation speed
    
analysis.get_predictions_for_bed_file("data/odd_training.h", "pred_results/") # two arguments: decompressed region file from "Data preparation" step as the input; output directory
```
The output files in the `Data preparation` step need to be decompressed before using as the input of this step. There are two output files from the prediction: 1) a `.tsv` file containing the probabilities of each distinct feature for each region; 2) a `.NA` file containing regions without predictions (should be empty). Repeat this step for each human and pig region file. Note that the prediction for large files is time-consuming, so users may consider break a large region file into small ones and use parallel computing.







