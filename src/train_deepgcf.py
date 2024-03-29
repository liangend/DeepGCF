'''
This script trains a single neural network for either hyper-parameter search or prediction.
'''
import sys,gzip,random,numpy as np,argparse,math
from sklearn.metrics import roc_curve, auc, average_precision_score, mean_squared_error
from shared_deepgcf import *

# Pytorch
import torch
import torch.nn as nn
import torch.utils.data

# Choose a random number of neurons
def randomNumNeurons():
    c = [8,16,32,64,128,256]
    return random.choice(c)

# Choose a random combination of hyper-parameters
def setTrainingHyperParameters():
    batch_size = random.choice([64,128,256,512,1024,2048,4096,8192])
    epsilon = random.choice([0.001,0.01,0.1]) # learning rate
    dropout_rate = random.choice([0,0.1,0.2,0.3,0.4,0.5])

    a = random.choice([1,2]) # number of species-specific hidden layers
    b = random.choice([1,2]) # number of hidden layers following the species-specific hidden layers
    num_layers = [a,a,b] # human-specific, pig-specific, and final
    num_neurons = [0,0,0,0,0,0]

    for i in range(a):
        num_neurons[i] = randomNumNeurons()
        num_neurons[i+2] = randomNumNeurons()
    for j in range(b):
        num_neurons[j+4]= randomNumNeurons()

    return batch_size,epsilon,dropout_rate,num_layers,num_neurons

# Train for one epoch
def train(human_data_filename,pig_data_filename,shuffled_pig_data_filename,
          neg_data_ratio,positive_training_data_size,num_batch,batch_size,num_features,
          optimizer,net):

    running_loss = 0.

    # Open all training data files
    with open(human_data_filename,'r') as positive_human_file,\
            open(pig_data_filename,'r') as positive_pig_file, \
            open(human_data_filename, 'r') as negative_human_file, \
            open(shuffled_pig_data_filename,'r') as negative_pig_file:
        next(positive_human_file)
        next(positive_pig_file)
        next(negative_human_file)
        next(negative_pig_file)
        # Iterate through each training batch
        for i in range(num_batch):
            # Current batch size kept the same in all batches except for the last one which may be smaller
            current_batch_size = batch_size if i<int(positive_training_data_size*2/batch_size) else (positive_training_data_size*2)%batch_size

            # Weight negative examples more by assigning smaller weight to positive samples
            # Training examples alternate between positive and negative
            weights = [1./neg_data_ratio,1] * int(current_batch_size/2)
            weights = torch.FloatTensor(weights)
            criterion = nn.BCELoss(weight=weights) # binary cross entropy loss

            # Read data for the current training batch
            ftr = [positive_human_file,positive_pig_file,negative_human_file,negative_pig_file]
            inputs,labels = readBatch(files_to_read=ftr,
                                      batch_size=current_batch_size,
                                      num_features=num_features)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = net(inputs) # forward
            loss = criterion(outputs,labels) # calculate BCE loss
            loss.backward() # backward
            optimizer.step() # optimize

            # Calculate loss
            running_loss += loss.data

            # Print progress
            printProgress(int((i+1)/num_batch*100),'Training')

    return net,running_loss

# Evaluate the current model with held-out validation data
def eval(human_data_filename,pig_data_filename,shuffled_pig_data_filename,num_batch,batch_size,positive_data_size,num_features,net):

    score_y = np.zeros((positive_data_size*2),dtype=float) # predictions
    true_y = np.zeros((positive_data_size*2), dtype=int) # labels
    true_y[::2] = 1 # odd examples are positive examples

    # Open all data files
    with open(human_data_filename,'r') as positive_human_file,\
            open(pig_data_filename,'r') as positive_pig_file, \
            open(human_data_filename, 'r') as negative_human_file, \
            open(shuffled_pig_data_filename,'r') as negative_pig_file:
        next(positive_human_file)
        next(positive_pig_file)
        next(negative_human_file)
        next(negative_pig_file)
        # Iterate through each batch
        for i in range(num_batch):
            # Current batch size kept the same in all batches except for the last one which may be smaller
            current_batch_size = batch_size if i < int(positive_data_size*2 / batch_size) else (positive_data_size*2) % batch_size

            # Read data for the current batch
            inputs,_ = readBatch(files_to_read=[positive_human_file,positive_pig_file,negative_human_file,negative_pig_file],
                                 batch_size=current_batch_size,
                                 num_features=num_features)

            # Make predictions using the current model
            score_y[i*batch_size:i*batch_size+current_batch_size] = net(inputs).data.numpy()

            # Print progress
            printProgress(int((i+1)/num_batch*100),'Evaluating')

    # Evaluate the predictions
    fpr, tpr, thresholds = roc_curve(true_y, score_y)
    current_auroc = auc(fpr, tpr) # AUROC
    current_auprc = average_precision_score(true_y, score_y) # AUPRC
    current_mse = mean_squared_error(true_y, score_y) # mean squared error
    pos_scores = score_y[::2]
    neg_scores = score_y[1::2]

    # Aggregate and return the evaluation result
    return [round(s,5) for s in [current_mse,current_auroc,current_auprc,np.mean(score_y),np.mean(pos_scores),np.mean(neg_scores)]]

# Read each training data file once and build a list of line offsets
def findLineOffsets(total_positive_training_data_size,
                    human_training_data_filename,pig_training_data_filename,shuffled_pig_training_data_filename):

    offsets = np.zeros((3,total_positive_training_data_size),dtype=int) # array of offsets

    offset = 0
    with open(human_training_data_filename,'r') as f:
        for i in range(total_positive_training_data_size):
            line = f.readline()
            offsets[0,i] = offset
            offset += len(line)

    offset = 0
    with open(pig_training_data_filename, 'r') as f:
        for i in range(total_positive_training_data_size):
            line = f.readline()
            offsets[1,i] = offset
            offset += len(line)

    offset = 0
    with open(shuffled_pig_training_data_filename, 'r') as f:
        for i in range(total_positive_training_data_size):
            line = f.readline()
            offsets[2,i] = offset
            offset += len(line)

    return offsets

def main():
    ### Input arguments start here ###
    parser = argparse.ArgumentParser(prog='python src/train_deepgcf.py', description='Train a neural network')

    parser.add_argument('-o', '--output-filename-prefix', help='output prefix (must be specified if saving (-v))',
                        type=str, default='tmp')
    parser.add_argument('-k', '--random-search', help='if hyper-parameters should be randomly set', action='store_true')
    parser.add_argument('-v', '--save', help='if the trained classifier should be saved after training',
                        action='store_true')
    parser.add_argument('-t', '--early-stopping',
                        help='if early stopping should be allowed (stopping before the maximum number of epochs if \
                        there is no improvement in validation AUROC in three consecutive epochs)',
                        action='store_true')
    parser.add_argument('-r', '--neg-data-ratio', help='weight ratio of negative samples to positive samples (default: 50)',
                        type=int, default=50)
    parser.add_argument('-s', '--seed', help='random seed (default: 1)', type=int, default=1)
    parser.add_argument('-e', '--num-epoch', help='maximum number of training epochs (default: 100)', type=int,
                        default=100)

    g1 = parser.add_argument_group('required arguments specifying training data')
    g1.add_argument('-A', '--human-training-data-filename', required=True, help='path to human training data file',
                    type=str)
    g1.add_argument('-B', '--pig-training-data-filename', required=True,
                    help='path to pig positive training data file', type=str)
    g1.add_argument('-C', '--shuffled-pig-training-data-filename', required=True,
                    help='path to pig shuffled/negative training data file', type=str)

    g1_2 = parser.add_argument_group('required arguments specifying validation data')
    g1_2.add_argument('-D', '--human-validation-data-filename', required=True,
                      help='path to human validation data file', type=str)
    g1_2.add_argument('-E', '--pig-validation-data-filename', required=True,
                      help='path to pig positive validation data file', type=str)
    g1_2.add_argument('-F', '--shuffled-pig-validation-data-filename', required=True,
                      help='path to pig shuffled/negative validation data file', type=str)

    g4 = parser.add_argument_group('required arguments describing feature data')
    g4.add_argument('-tr', '--positive-training-data-size',
                    help='number of samples in positive training data to *use* (default: 1000000)', type=int,
                    default=1000000)
    g4.add_argument('-tra', '--total-positive-training-data-size',
                    help='number of samples in total positive training data to *read*', type=int)
    g4.add_argument('-va', '--positive-validation-data-size',
                    help='number of samples in positive validation data to use (default: 100000)', type=int,
                    default=100000)
    g4.add_argument('-hf', '--num-human-features',
                    help='number of human features in input vector (default: 8824)', type=int, default=8824)
    g4.add_argument('-mf', '--num-pig-features',
                    help='number of pig features in input vector (default: 3113)', type=int, default=3113)


    g2 = parser.add_argument_group(
        'optional arguments specifying hyper-parameters (ignored if random search (-k) is specified)')
    g2.add_argument('-b', '--batch-size', help='batch size (default: 128)', type=int, default=128)
    g2.add_argument('-l', '--learning-rate', help='epsilon (default: 0.1)', type=float, default=0.1)
    g2.add_argument('-d', '--dropout-rate', help='dropout rate (default: 0.1)', type=float, default=0.1)

    g2.add_argument('-nl1', '--num-layers-1', help='number of hidden layers in species-specific sub-networks (default: 1)',
                    type=int, default=1)
    g2.add_argument('-nl2', '--num-layers-2', help='number of hidden layers in final sub-network (default: 1)', type=int,
                    default=1)
    g2.add_argument('-nnh1', '--num-neuron-human-1',
                    help='number of neurons in the first hidden layer in the human-specific sub-network (default :1)', type=int,
                    default=256)
    g2.add_argument('-nnh2', '--num-neuron-human-2',
                    help='number of neurons in the second hidden layer in the human-specific sub-network (default: 0)', type=int,
                    default=0)
    g2.add_argument('-nnm1', '--num-neuron-pig-1',
                    help='number of neurons in the first hidden layer in the pig-specific sub-network (default: 128)', type=int,
                    default=128)
    g2.add_argument('-nnm2', '--num-neuron-pig-2',
                    help='number of neurons in the second hidden layer in the pig-specific sub-network (default: 0)', type=int,
                    default=0)
    g2.add_argument('-nn1', '--num-neuron-1',
                    help='number of neurons in the first hidden layer in the final sub-network (default: 256)', type=int,
                    default=256)
    g2.add_argument('-nn2', '--num-neuron-2',
                    help='number of neurons in the second hidden layer in the final sub-network (default: 0)', type=int,
                    default=0)

    args = parser.parse_args()
    ### Input arguments end here ###



    ### Set-up before training starts here ###
    # Set all random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Randomly set hyper-parameters if needed
    if args.random_search:
        batch_size,learning_rate,dropout_rate,num_layers,num_neurons = setTrainingHyperParameters()
    else:
        batch_size,learning_rate,dropout_rate = args.batch_size,args.learning_rate,args.dropout_rate
        num_layers = [args.num_layers_1,args.num_layers_1,args.num_layers_2]
        num_neurons = [args.num_neuron_human_1,args.num_neuron_human_2,args.num_neuron_pig_1,args.num_neuron_pig_2,args.num_neuron_1,args.num_neuron_2]

    # Store and print hyper-parameters
    hyperparameters = [args.seed,args.neg_data_ratio,
                       args.positive_training_data_size,args.positive_training_data_size,
                       args.positive_validation_data_size,args.positive_validation_data_size,
                       batch_size,learning_rate,dropout_rate]+num_layers+num_neurons
    hyperparameter_names = ['Seed',
                            'Negative data ratio',
                            'Positive training data size',
                            'Negative training data size',
                            'Positive validation data size',
                            'Negative validation data size',
                            'Batch size',
                            'Epsilon (learning rate)',
                            'Dropout rate',
                            'Number of layers in human-specific network',
                            'Number of layers in pig-specific network',
                            'Number of layers in final network',
                            'Number of neurons in the 1st layer of human-specific network',
                            'Number of neurons in the 2nd layer of human-specific network',
                            'Number of neurons in the 1st layer of pig-specific network',
                            'Number of neurons in the 2nd layer of pig-specific network',
                            'Number of neurons in the 1st layer of final network',
                            'Number of neurons in the 2nd layer of final network']
    for i in range(len(hyperparameters)):
        print (hyperparameter_names[i]+':',hyperparameters[i])


    # If we are sampling from a large training dataset, we read the files once to build a list of line offsets
    # This way we can skip lines that we do not need to read
    if args.total_positive_training_data_size>args.positive_training_data_size:
        print ('Reading training data files once to build a list of line offsets...')
        offsets = findLineOffsets(args.total_positive_training_data_size,
                                  args.human_training_data_filename,args.pig_training_data_filename,args.shuffled_pig_training_data_filename)

        # Select lines from the training data to use
        print ('Selecting lines to read for training data...')
        lines_to_read = np.zeros((4,args.positive_training_data_size),dtype=int)
        positive_samples_to_read = sorted(random.sample(range(args.total_positive_training_data_size),k=args.positive_training_data_size))
        print (positive_samples_to_read[:10])
        lines_to_read[0,:] = offsets[0,positive_samples_to_read] # Positive human training data
        lines_to_read[1,:] = offsets[1,positive_samples_to_read] # Positive pig training data
        lines_to_read[2,:] = sorted(random.sample(list(offsets[0,:]),k=args.positive_training_data_size)) # Negative human training data
        lines_to_read[3,:] = sorted(random.sample(list(offsets[2,:]),k=args.positive_training_data_size)) # Negative pig training data
    else:
        lines_to_read = -1

    # Build neural network
    net = PseudoSiameseNet(args.num_human_features, args.num_pig_features, num_layers, num_neurons, dropout_rate)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    # Print header to for training results
    header = ['seed','nr','trainp','trainn','valp','valn','batch','eps','dropout','l','l','l','n','n','n','n','n','n',
                  'epoch','loss','tmse','tauroc','tauprc','tmean','tmeanp','tmeann','vmse','vauroc','vauprc','vmean','vmeanp','vmeann']
    print('\t'.join(header))

    # Calculate total number of batches
    num_training_batch = int(math.ceil(args.positive_training_data_size * 2 / batch_size))
    num_validation_batch = int(math.ceil(args.positive_validation_data_size * 2 / batch_size))
    ### Set-up before training ends here ###





    ####### Training starts here #######
    prev_aurocs = []
    prev_results= []
    for epoch in range(args.num_epoch): # loop over the training data multiple times

        # Train
        net.train() # set neural network to train mode
        net,running_loss = train(args.human_training_data_filename,
                                 args.pig_training_data_filename,
                                 args.shuffled_pig_training_data_filename,
                                 args.neg_data_ratio,
                                 args.positive_training_data_size,
                                 num_training_batch,
                                 batch_size,
                                 [args.num_human_features,args.num_pig_features],
                                 optimizer,
                                 net)
        running_loss /= num_training_batch

        net.eval() # set neural network to evaluation mode

        # Evaluate using training data to check overfitting
        training_result = eval(args.human_training_data_filename,
                               args.pig_training_data_filename,
                               args.shuffled_pig_training_data_filename,
                               num_training_batch,
                               batch_size,
                               args.positive_training_data_size,
                               [args.num_human_features,args.num_pig_features],
                               net)

        # Evaluate using validation data
        validation_result = eval(args.human_validation_data_filename,
                                 args.pig_validation_data_filename,
                                 args.shuffled_pig_validation_data_filename,
                                 num_validation_batch,
                                 batch_size,
                                 args.positive_validation_data_size,
                                 [args.num_human_features,args.num_pig_features],
                                 net)

        current_auroc = validation_result[1] # current validation AUROC
        result = hyperparameters+[epoch,np.round(running_loss,5)]+training_result+validation_result # save all results

        # Print this epoch's result
        sys.stdout.write("\r"+" "*200)
        sys.stdout.write('\r'+'\t'.join([str(s) for s in result])+'\n')
        sys.stdout.flush()

        # Save the current epoch's model if it has the highest validation AUROC and if we want to save it
        if args.save and (epoch==0 or (len(prev_aurocs)>0 and current_auroc>max(prev_aurocs))):
            fn = '_'.join([args.output_filename_prefix]+[str(s) for s in hyperparameters])+'.pt' # filename
            torch.save(net,fn)

        # Stop if there is no improvement in AUROC over the last three training epochs and if early stopping is allowed
        elif args.early_stopping and len(prev_aurocs)>=2 and current_auroc<=prev_aurocs[-1]<=prev_aurocs[-2]:
            print ('Early stopping due to no improvement in AUROC in the last 3 epochs')
            break

        prev_aurocs.append(current_auroc) # save current validation AUROC
        prev_results.append(result)
    ####### Training ends here #######





    # Report the best performing epoch's results
    best_performing_epoch = prev_aurocs.index(max(prev_aurocs))
    print ('\nBest performing epoch: %d' % best_performing_epoch)
    print ('\t'.join([str(s) for s in prev_results[best_performing_epoch]]))

main()
