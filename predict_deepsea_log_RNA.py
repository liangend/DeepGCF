'''
This script generates predictions for a set of pairs given a trained neural network.
'''
import gzip,random,numpy as np,argparse

# Pytorch
import torch
from torch.autograd import Variable
import torch.utils.data
from shared_deepsea_log_RNA import *

#hbf = open("/Users/lijinghui/Desktop/odd_validation_test.h.tsv", "r")
#mbf = open("/Users/lijinghui/Desktop/odd_validation_test.m.tsv", "r")
#hrf = gzip.open("/Users/lijinghui/Desktop/odd_validation.h.gz",'rb')
#mrf = gzip.open("/Users/lijinghui/Desktop/odd_validation.m.gz",'rb')
#next(hbf)
#next(mbf)

#num_features = [309,294,862,578,77,80]
#current_batch_size = 16
#rnaseq_range = [[0, 72071],[0, 264427]]

def predict(net,
            human_test_binary_filename,mouse_test_binary_filename,
            human_test_rna_filename,mouse_test_rna_filename,output_filename,
            test_data_size,batch_size,
            num_features,
            human_rnaseq_range,mouse_rnaseq_range):

    # Difference between maximum and minimum RNA-seq signals
    human_rnaseq_range_log = []
    human_rnaseq_range_log.append(np.log(human_rnaseq_range[0] + 0.00001))
    human_rnaseq_range_log.append(np.log(human_rnaseq_range[1]))
    mouse_rnaseq_range_log = []
    mouse_rnaseq_range_log.append(np.log(mouse_rnaseq_range[0] + 0.00001))
    mouse_rnaseq_range_log.append(np.log(mouse_rnaseq_range[1]))

    hrr = human_rnaseq_range_log[1]-human_rnaseq_range_log[0]
    mrr = mouse_rnaseq_range_log[1]-mouse_rnaseq_range_log[0]

    # Make predictions and write to output
    with open(human_test_binary_filename,'r') as hbf,\
            open(mouse_test_binary_filename,'r') as mbf,\
            gzip.open(human_test_rna_filename,'rb') as hrf,\
            gzip.open(mouse_test_rna_filename,'rb') as mrf,\
            gzip.open(output_filename if output_filename.endswith('.gz') else output_filename+'.gz','wb') as fout:
        next(hbf)
        next(mbf)
        for i in range(int(test_data_size/batch_size)+1): # iterate through each batch
            current_batch_size = batch_size if i<int(test_data_size/batch_size) else test_data_size%batch_size

            if current_batch_size==0:
                break

            X = np.zeros((current_batch_size,num_features[0]+num_features[4]),dtype=float) # to store human data
            Y = np.zeros((current_batch_size,num_features[1]+num_features[5]),dtype=float) # to store mouse data

            for j in range(current_batch_size): # iterate through each sample within the batch
                ### Read binary features
                # Read human binary features
                hb = hbf.readline().strip().split('\t')
                del hb[0:6]
                for k in range(0, len(hb)):
                    hb[k] = float(hb[k])

                # Read mouse binary features
                mb = mbf.readline().strip().split('\t')
                del mb[0:6]
                for k in range(0, len(mb)):
                    mb[k] = float(mb[k])
                
                ### Read RNA features
                hl = hrf.readline().strip().split(b'|')
                ml = mrf.readline().strip().split(b'|')

                # Indices of the non-zero feature indices
                nonzero_human_feature_indices = [int(s) for s in hl[1].strip().split()]
                nonzero_mouse_feature_indices = [int(s) for s in ml[1].strip().split()]
                human_rna_indices = [x for x in nonzero_human_feature_indices 
                                      if x >= num_features[2] - num_features[4]]
                mouse_rna_indices = [x for x in nonzero_mouse_feature_indices 
                                      if x >= num_features[3] - num_features[5]]

                # Normalize RNA-seq values
                human_rna = [(np.log(float(s) + 0.00001) - human_rnaseq_range_log[0])/hrr
                            for s in hl[2].strip().split()] if len(hl)>1 else []
                
                mouse_rna = [(np.log(float(s) + 0.00001) - mouse_rnaseq_range_log[0])/mrr
                            for s in ml[2].strip().split()] if len(ml)>1 else []
                
                real_valued_human_features = [0] * num_features[4]
                real_valued_mouse_features = [0] * num_features[5]
                if len(human_rna_indices) > 0:
                    human_rna_indices = [x - (num_features[2] - num_features[4]) 
                                        for x in human_rna_indices]
                    human_rna = human_rna[(len(human_rna)-len(human_rna_indices)):
                                          len(human_rna)]
                    for k in range(len(human_rna_indices)):
                        real_valued_human_features[human_rna_indices[k]] = human_rna[k]

                if len(mouse_rna_indices) > 0:
                    mouse_rna_indices = [x - (num_features[3] - num_features[5]) 
                                                 for x in mouse_rna_indices]
                    mouse_rna = human_rna[(len(mouse_rna)-len(mouse_rna_indices)):
                                          len(mouse_rna)]
                    for k in range(len(mouse_rna_indices)):
                        real_valued_mouse_features[mouse_rna_indices[k]] = mouse_rna[k]
                # Set non-zero features to the corresponding values
                h = np.concatenate((hb,real_valued_human_features))
                m = np.concatenate((mb,real_valued_mouse_features))
                
                X[j, :] = h
                Y[j, :] = m

            # Convert feature matrices for PyTorch
            X = Variable(torch.from_numpy(X).float())
            Y = Variable(torch.from_numpy(Y).float())
            inputs = torch.cat((X,Y),1) # concatenate human and mouse data

            # Make prediction on current batch
            y_pred = net(inputs) # put the feature matrix into the provided trained PSNN
            y_pred = y_pred.data

            # Write predicted probabilities of the current batch
            sample_output = [str(np.round(y_pred[j],7)) for j in range(current_batch_size)]
            l = '\n'.join(sample_output)+'\n'
            fout.write(l.encode())

def main():
    epilog = '# Example: python source/predict.py -t NN/odd_ensemble/NN_1_*.pt -H data/even_all.h.gz -M \
    data/even_all.m.gz -d 16627449 -o NN/odd_ensemble/even_all_1.gz'

    parser = argparse.ArgumentParser(prog='python source/predict.py',
                                     description='Generate predictions given a trained neural network',
                                     epilog=epilog)
    parser.add_argument('-s', '--seed', help='random seed (default: 1)', type=int, default=1)
    parser.add_argument('-b', '--batch-size', help='batch size (default: 128)', type=int, default=128)

    g1 = parser.add_argument_group('required arguments specifying input and output')
    g1.add_argument('-t', '--trained-classifier-filename', required=True, help='path to a trained classifier (.pt)',
                    type=str)
    g1.add_argument('-Hb', '--human-binary-filename', required=True, help='path to human binary feature data file', type=str)
    g1.add_argument('-Mb', '--mouse-binary-filename', required=True, help='path to mouse binary feature data file', type=str)
    g1.add_argument('-Hr', '--human-rna-filename', required=True, help='path to human RNA feature data file', type=str)
    g1.add_argument('-Mr', '--mouse-rna-filename', required=True, help='path to mouse RNA feature data file', type=str)
    g1.add_argument('-d', '--data-size', required=True, help='number of samples', type=int)
    g1.add_argument('-o', '--output-filename', required=True, help='path to output file', type=str)

    g1.add_argument('-hbf', '--num-human-binary-features',
                    help='number of human binary features in input vector (default: 8824)', type=int, default=8824)
    g1.add_argument('-mbf', '--num-mouse-binary-features',
                    help='number of mouse binary features in input vector (default: 3113)', type=int, default=3113)
    g1.add_argument('-hf', '--num-human-ori-features',
                    help='number of human original features in input vector (default: 8824)', type=int, default=8824)
    g1.add_argument('-mf', '--num-mouse-ori-features',
                    help='number of mouse original features in input vector (default: 3113)', type=int, default=3113)
    g1.add_argument('-hrf', '--num-human-rna-features',
                    help='number of human RNA features in input vector (default: 8824)', type=int, default=8824)
    g1.add_argument('-mrf', '--num-mouse-rna-features',
                    help='number of mouse RNA features in input vector (default: 3113)', type=int, default=3113)
    g1.add_argument('-hrmin', '--human-rnaseq-min',
                    help='minimum expression level in human RNA-seq data (default: 8e-05)', type=float, default=8e-05)
    g1.add_argument('-hrmax', '--human-rnaseq-max',
                    help='maximum expression level in human RNA-seq data (default: 1.11729e06)', type=float, default=1.11729e06)
    g1.add_argument('-mrmin', '--mouse-rnaseq-min',
                    help='minimum expression level in mouse RNA-seq data (default: 0.00013)', type=float, default=0.00013)
    g1.add_argument('-mrmax', '--mouse-rnaseq-max',
                    help='maximum expression level in mouse RNA-seq data (default: 41195.3)', type=float, default=41195.3)
    
    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load previously trained classifier
    net = torch.load(args.trained_classifier_filename)
    net.eval() # make sure it's in evaluation mode

    # Number of features
    num_features = [args.num_human_binary_features,args.num_mouse_binary_features,
                        args.num_human_ori_features,args.num_mouse_ori_features,
                        args.num_human_rna_features,args.num_mouse_rna_features]
    # Make predictions
    predict(net,
            args.human_binary_filename,args.mouse_binary_filename,
            args.human_rna_filename,args.mouse_rna_filename,args.output_filename,
            args.data_size,args.batch_size,
            num_features,
            [args.human_rnaseq_min,args.human_rnaseq_max],
            [args.mouse_rnaseq_min,args.mouse_rnaseq_max])

if __name__ == "__main__":
    main()
