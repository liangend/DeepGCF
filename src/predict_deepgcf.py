'''
This script generates predictions for a set of pairs given a trained neural network.
'''
import gzip,random,numpy as np,argparse

# Pytorch
import torch
from torch.autograd import Variable
import torch.utils.data
from shared_deepgcf import *

#hf = open("/Users/lijinghui/Desktop/test/shuf_hg.tsv", "r")
#next(hf)
#num_human_features = 294
#batch_size = 2
#test_data_size = 10
#i = 0
#j = 0

def predict(net,
            human_test_filename,pig_test_filename,output_filename,
            test_data_size,batch_size,
            num_human_features,num_pig_features):

    # Make predictions and write to output
    with open(human_test_filename,'r') as hf,\
            open(pig_test_filename,'r') as mf,\
            gzip.open(output_filename if output_filename.endswith('.gz') else output_filename+'.gz','wb') as fout:
        next(hf)
        next(mf)
        for i in range(int(test_data_size/batch_size)+1): # iterate through each batch
            current_batch_size = batch_size if i<int(test_data_size/batch_size) else test_data_size%batch_size

            if current_batch_size==0:
                break

            X = np.zeros((current_batch_size,num_human_features),dtype=float) # to store human data
            Y = np.zeros((current_batch_size,num_pig_features),dtype=float) # to store pig data

            for j in range(current_batch_size): # iterate through each sample within the batch
                hl = hf.readline().strip().split('\t')
                del hl[0:6]
                for k in range(0, len(hl)):
                    hl[k] = float(hl[k])

                ml = mf.readline().strip().split('\t')
                del ml[0:6]
                for k in range(0, len(ml)):
                    ml[k] = float(ml[k])
                
                X[j, :] = hl
                Y[j, :] = ml

            # Convert feature matrices for PyTorch
            X = Variable(torch.from_numpy(X).float())
            Y = Variable(torch.from_numpy(Y).float())
            inputs = torch.cat((X,Y),1) # concatenate human and pig data

            # Make prediction on current batch
            y_pred = net(inputs) # put the feature matrix into the provided trained PSNN
            y_pred = y_pred.data

            # Write predicted probabilities of the current batch
            sample_output = [str(np.round(y_pred[j],7)) for j in range(current_batch_size)]
            l = '\n'.join(sample_output)+'\n'
            fout.write(l.encode())

def main():
    epilog = '# Example: python src/predict_deepgcf.py -t NN/odd_ensemble/NN_1_*.pt -H data/even_all.h.gz -M \
    data/even_all.m.gz -d 16627449 -o NN/odd_ensemble/even_all_1.gz'

    parser = argparse.ArgumentParser(prog='python src/predict_deepgcf.py',
                                     description='Generate predictions given a trained neural network',
                                     epilog=epilog)
    parser.add_argument('-s', '--seed', help='random seed (default: 1)', type=int, default=1)
    parser.add_argument('-b', '--batch-size', help='batch size (default: 128)', type=int, default=128)

    g1 = parser.add_argument_group('required arguments specifying input and output')
    g1.add_argument('-t', '--trained-classifier-filename', required=True, help='path to a trained classifier (.pt)',
                    type=str)
    g1.add_argument('-H', '--human-feature-filename', required=True, help='path to human feature data file', type=str)
    g1.add_argument('-M', '--pig-feature-filename', required=True, help='path to pig feature data file', type=str)
    g1.add_argument('-d', '--data-size', required=True, help='number of samples', type=int)
    g1.add_argument('-o', '--output-filename', required=True, help='path to output file', type=str)

    g1.add_argument('-hf', '--num-human-features',
                    help='number of human features in input vector (default: 8824)', type=int, default=8824)
    g1.add_argument('-mf', '--num-pig-features',
                    help='number of pig features in input vector (default: 3113)', type=int, default=3113)

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load previously trained classifier
    net = torch.load(args.trained_classifier_filename)
    net.eval() # make sure it's in evaluation mode

    # Make predictions
    predict(net,
            args.human_feature_filename,args.pig_feature_filename,args.output_filename,
            args.data_size,args.batch_size,
            args.num_human_features,args.num_pig_features)

if __name__ == "__main__":
    main()
