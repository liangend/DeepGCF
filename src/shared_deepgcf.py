'''
This script defines the neural network architecture and functions useful in training
'''

import sys,scipy.sparse,numpy as np,torch

# Print progress given a percentage and appropriate message
def printProgress(p,messsage):
    sys.stdout.write("\r[%s%s] %d%% %s    " % ("=" * p, " " * (100 - p), p, messsage))
    sys.stdout.flush()

#positive_human_file = open("/Users/lijinghui/Desktop/test/shuf_hg.tsv", "r")
#positive_pig_file = open("/Users/lijinghui/Desktop/test/shuf_sus.tsv", "r")
#negative_human_file = open("/Users/lijinghui/Desktop/test/shuf_hg.tsv", "r")
#negative_pig_file = open("/Users/lijinghui/Desktop/test/shufx2_sus.tsv", "r")
#next(positive_human_file)
#next(positive_pig_file)
#next(negative_human_file)
#next(negative_pig_file)
#files_to_read = [positive_human_file, positive_pig_file, negative_human_file, negative_pig_file]
#num_features = [294,294]
#batch_size = 2

def readBatch(files_to_read,batch_size,num_features):
    # File pointers
    positive_human_data_file = files_to_read[0]
    positive_pig_data_file = files_to_read[1]
    negative_human_data_file = files_to_read[2]
    negative_pig_data_file = files_to_read[3]

    # Lists needed to construct a SciPy sparse matrix
    row,col,data = [],[],[] # row indices, column indices, feature values

    i = 0 # example index
    while i<batch_size:
        ### Read one positive example
        hl_pos = positive_human_data_file.readline().strip().split('\t')
        del hl_pos[0:6]
        for j in range(0, len(hl_pos)):
            hl_pos[j] = float(hl_pos[j])

        ml_pos = positive_pig_data_file.readline().strip().split('\t')
        del ml_pos[0:6]
        for j in range(0, len(ml_pos)):
            ml_pos[j] = float(ml_pos[j])
        
        ### Read one negative example
        hl_neg = negative_human_data_file.readline().strip().split('\t')
        del hl_neg[0:6]
        for j in range(0, len(hl_neg)):
            hl_neg[j] = float(hl_neg[j])
        
        ml_neg = negative_pig_data_file.readline().strip().split('\t')
        del ml_neg[0:6]
        for j in range(0, len(ml_neg)):
            ml_neg[j] = float(ml_neg[j])

        ### Save data for the two examples
        row += [i]*(len(hl_pos) + len(ml_pos)) + \
               [i+1]*(len(hl_neg) +len(ml_neg))
        col += list(range(0, num_features[0]+num_features[1])) + \
            list(range(0, num_features[0]+num_features[1]))
        data += hl_pos + ml_pos + hl_neg + ml_neg

        i += 2 # read two examples in the while loop, one positive example and one negative example

    # Build a SciPy sparse matrix with feature data and convert it into an array
    X = scipy.sparse.coo_matrix((data,(row,col)),shape=(batch_size,num_features[0]+num_features[1])).toarray()

    # Build a label array
    Y = np.zeros(batch_size,dtype=int) # label
    Y[::2] = 1 # odd examples are positive examples

    # Convert data for PyTorch
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float()
    X,Y = torch.autograd.Variable(X), torch.autograd.Variable(Y)
    return X,Y

# A pseudo-Siamese neural network
class PseudoSiameseNet(torch.nn.Module):
    def __init__(self,num_human_features,num_pig_features,num_layers,num_neurons,dropout_rate):
        """
        num_layer_species: number of layers in species-specific sub-networks
        num_layer_final: number of layers in final sub-network
        num_neuron_human: number of neurons in each layer of human-specific sub-network
        num_neuron_pig: number of neurons in each layer of pig-specific sub-network
        num_neuron_final: number of neurons in each layer of final sub-network
        """
        super(PseudoSiameseNet, self).__init__()
        self.num_human_features = num_human_features
        self.num_layer_species = num_layers[0]
        self.num_layer_final = num_layers[2]
        self.num_neuron_human = num_neurons[:self.num_layer_species]
        self.num_neuron_pig = num_neurons[2:2+self.num_layer_species]
        self.num_neuron_final = num_neurons[4:4+self.num_layer_final]

        # Sequence of operations done only on either human or pig features
        self.human_layers = torch.nn.Sequential()
        self.pig_layers = torch.nn.Sequential()
        for i in range(self.num_layer_species):
            if i==0: # from input features to first hidden layer
                self.human_layers.add_module('h'+str(i),
                                             torch.nn.Linear(num_human_features,int(self.num_neuron_human[i]),bias=False))
                self.pig_layers.add_module('m'+str(i),
                                             torch.nn.Linear(num_pig_features,int(self.num_neuron_pig[i]),bias=False))
            else:
                self.human_layers.add_module('h'+str(i),
                                             torch.nn.Linear(int(self.num_neuron_human[i-1]),int(self.num_neuron_human[i])))
                self.pig_layers.add_module('m'+str(i),
                                             torch.nn.Linear(int(self.num_neuron_pig[i-1]),int(self.num_neuron_pig[i])))
            self.human_layers.add_module('h'+str(i)+'dropout',torch.nn.Dropout(p=dropout_rate)) # dropout
            self.human_layers.add_module('hr'+str(i),torch.nn.ReLU()) # relu
            self.pig_layers.add_module('m'+str(i)+'dropout',torch.nn.Dropout(p=dropout_rate)) # dropout
            self.pig_layers.add_module('mr'+str(i),torch.nn.ReLU()) # relu

        # Sequence of operations done on concatenated output of species-specific sub-networks
        self.final_layers = torch.nn.Sequential()
        for i in range(self.num_layer_final):
            if i==0: # from concatenated output of species-specific sub-networks to the first layer of final sub-network
                self.final_layers.add_module('c'+str(i),
                                             torch.nn.Linear(int(self.num_neuron_human[-1])+int(self.num_neuron_pig[-1]),int(self.num_neuron_final[i])))
            else:
                self.final_layers.add_module('c'+str(i),
                                             torch.nn.Linear(int(self.num_neuron_final[i-1]),int(self.num_neuron_final[i])))
            self.final_layers.add_module('cd'+str(i),torch.nn.Dropout(p=dropout_rate))
            self.final_layers.add_module('cr'+str(i),torch.nn.ReLU())

        # Output layer
        self.final_layers.add_module('end',torch.nn.Linear(int(self.num_neuron_final[-1]),1)) # from last layer to single output
        self.final_layers.add_module('sigmoid',torch.nn.Sigmoid())

    def forward(self,x):
        h = self.human_layers.forward(x[:,:self.num_human_features]) # human-specific sub-network
        m = self.pig_layers.forward(x[:,self.num_human_features:]) # pig-specific sub-network
        c = torch.cat((h,m),1) # concatenate the output from human-specific sub-network and pig-specific sub-network
        y = self.final_layers.forward(c) # final/final sub-network
        y = y.view(c.size()[0])
        return y
