'''
This script defines the neural network architecture and functions useful in training
'''

import sys,scipy.sparse,numpy as np,torch

# Print progress given a percentage and appropriate message
def printProgress(p,messsage):
    sys.stdout.write("\r[%s%s] %d%% %s    " % ("=" * p, " " * (100 - p), p, messsage))
    sys.stdout.flush()

positive_human_file = open("/Users/lijinghui/Desktop/odd_validation_test.h.tsv", "r")
positive_mouse_file = open("/Users/lijinghui/Desktop/odd_validation_test.m.tsv", "r")
negative_human_file = open("/Users/lijinghui/Desktop/odd_validation_test.h.tsv", "r")
negative_mouse_file = open("/Users/lijinghui/Desktop/odd_validation_test.m.tsv", "r")
positive_human_RNA = gzip.open("/Users/lijinghui/Desktop/odd_validation.h.gz",'rb')
positive_mouse_RNA = gzip.open("/Users/lijinghui/Desktop/odd_validation.m.gz",'rb')
negative_human_RNA = gzip.open("/Users/lijinghui/Desktop/odd_validation.h.gz",'rb')
negative_mouse_RNA = gzip.open("/Users/lijinghui/Desktop/odd_validation.m.gz",'rb')
next(positive_human_file)
next(positive_mouse_file)
next(negative_human_file)
next(negative_mouse_file)
files_to_read = [positive_human_file, positive_mouse_file, negative_human_file, negative_mouse_file,
                positive_human_RNA, positive_mouse_RNA, negative_human_RNA, negative_mouse_RNA]

num_features = [309,294,862,578,77,80]
batch_size = 16
rnaseq_range = [[0, 72071],[0, 264427]]

def readBatch(files_to_read,batch_size,num_features,rnaseq_range):
    # File pointers
    positive_human_data_file = files_to_read[0]
    positive_mouse_data_file = files_to_read[1]
    negative_human_data_file = files_to_read[2]
    negative_mouse_data_file = files_to_read[3]
    positive_human_RNA = files_to_read[4]
    positive_mouse_RNA = files_to_read[5]
    negative_human_RNA = files_to_read[6]
    negative_mouse_RNA = files_to_read[7]

    # Store RNA-seq ranges for normalization
    human_rnaseq_range,mouse_rnaseq_range = rnaseq_range[0],rnaseq_range[1]
    human_rnaseq_range_log = []
    human_rnaseq_range_log.append(np.log(human_rnaseq_range[0] + 0.00001))
    human_rnaseq_range_log.append(np.log(human_rnaseq_range[1]))
    mouse_rnaseq_range_log = []
    mouse_rnaseq_range_log.append(np.log(mouse_rnaseq_range[0] + 0.00001))
    mouse_rnaseq_range_log.append(np.log(mouse_rnaseq_range[1]))

    # Difference between maximum and minimum RNA-seq signal values
    hrr,mrr = human_rnaseq_range_log[1]-human_rnaseq_range_log[0], mouse_rnaseq_range_log[1]-mouse_rnaseq_range_log[0]

    # Lists needed to construct a SciPy sparse matrix
    row,col,data = [],[],[] # row indices, column indices, feature values

    i = 0 # example index
    while i<batch_size:
        ##### Read binary data
        ### Read one positive example
        hl_pos = positive_human_data_file.readline().strip().split('\t')
        del hl_pos[0:6]
        for j in range(0, len(hl_pos)):
            hl_pos[j] = float(hl_pos[j])

        ml_pos = positive_mouse_data_file.readline().strip().split('\t')
        del ml_pos[0:6]
        for j in range(0, len(ml_pos)):
            ml_pos[j] = float(ml_pos[j])
        
        ### Read one negative example
        hl_neg = negative_human_data_file.readline().strip().split('\t')
        del hl_neg[0:6]
        for j in range(0, len(hl_neg)):
            hl_neg[j] = float(hl_neg[j])
        
        ml_neg = negative_mouse_data_file.readline().strip().split('\t')
        del ml_neg[0:6]
        for j in range(0, len(ml_neg)):
            ml_neg[j] = float(ml_neg[j])

        ##### Read RNAseq data
        ### Read one positive example
        hl = positive_human_RNA.readline().strip().split(b'|')
        ml = positive_mouse_RNA.readline().strip().split(b'|')
        positive_nonzero_human_feature_indices = [int(s) for s in hl[1].strip().split()]
        positive_human_rna_indices = [x for x in positive_nonzero_human_feature_indices 
                                      if x >= num_features[2] - num_features[4]]
        positive_nonzero_mouse_feature_indices = [int(s) for s in ml[1].strip().split()]
        positive_mouse_rna_indices = [x for x in positive_nonzero_mouse_feature_indices 
                                      if x >= num_features[3] - num_features[5]]
        
        # Normalize RNA-seq values for the positive example
        positive_human_rna = [(np.log(float(s) + 0.00001) - human_rnaseq_range_log[0])/hrr
                                               for s in hl[2].strip().split()] if len(hl)>1 else []
        positive_mouse_rna = [(np.log(float(s) + 0.00001) - mouse_rnaseq_range_log[0])/mrr
                                               for s in ml[2].strip().split()] if len(ml)>1 else []
        positive_real_valued_human_features = [0] * num_features[4]
        positive_real_valued_mouse_features = [0] * num_features[5]

        if len(positive_human_rna_indices) > 0:
            positive_human_rna_indices = [x - (num_features[2] - num_features[4])
                                      for x in positive_human_rna_indices]
            positive_human_rna = positive_human_rna[(len(positive_human_rna) - len(positive_human_rna_indices)):
                                                    len(positive_human_rna)]
            for k in range(len(positive_human_rna_indices)):
                positive_real_valued_human_features[positive_human_rna_indices[k]] = positive_human_rna[k]

        if len(positive_mouse_rna_indices) > 0:
            positive_mouse_rna_indices = [x - (num_features[3] - num_features[5])
                                      for x in positive_mouse_rna_indices]
            positive_mouse_rna = positive_mouse_rna[(len(positive_mouse_rna) - len(positive_mouse_rna_indices)):
                                                    len(positive_mouse_rna)]
            for k in range(len(positive_mouse_rna_indices)):
                positive_real_valued_mouse_features[positive_mouse_rna_indices[k]] = positive_mouse_rna[k]

        ### Read one negative example
        hl = negative_human_RNA.readline().strip().split(b'|')
        ml = negative_mouse_RNA.readline().strip().split(b'|')
        negative_nonzero_human_feature_indices = [int(s) for s in hl[1].strip().split()]
        negative_human_rna_indices = [x for x in negative_nonzero_human_feature_indices 
                                      if x >= num_features[2] - num_features[4]]
        negative_nonzero_mouse_feature_indices = [int(s) for s in ml[1].strip().split()]
        negative_mouse_rna_indices = [x for x in negative_nonzero_mouse_feature_indices 
                                      if x >= num_features[3] - num_features[5]]

        # Normalize RNA-seq values for the negative example
        negative_human_rna = [(np.log(float(s) + 0.00001) - human_rnaseq_range_log[0])/hrr
                                               for s in hl[2].strip().split()] if len(hl)>1 else []
        negative_mouse_rna = [(np.log(float(s) + 0.00001) - mouse_rnaseq_range_log[0])/mrr
                                               for s in ml[2].strip().split()] if len(ml)>1 else []

        negative_real_valued_human_features = [0] * num_features[4]
        negative_real_valued_mouse_features = [0] * num_features[5]

        if len(negative_human_rna_indices) > 0:
            negative_human_rna_indices = [x - (num_features[2] - num_features[4]) 
                                      for x in negative_human_rna_indices]
            negative_human_rna = negative_human_rna[(len(negative_human_rna) - len(negative_human_rna_indices)):
                                                    len(negative_human_rna)]
            for k in range(len(negative_human_rna_indices)):
                negative_real_valued_human_features[negative_human_rna_indices[k]] = negative_human_rna[k]

        if len(negative_mouse_rna_indices) > 0:
            negative_mouse_rna_indices = [x - (num_features[3] - num_features[5]) 
                                      for x in negative_mouse_rna_indices]
            negative_mouse_rna = negative_mouse_rna[(len(negative_mouse_rna) - len(negative_mouse_rna_indices)):
                                                    len(negative_mouse_rna)]
            for k in range(len(negative_mouse_rna_indices)):
                negative_real_valued_mouse_features[negative_mouse_rna_indices[k]] = negative_mouse_rna[k]

        ### Save data for the two examples
        row += [i]*(num_features[0]+num_features[1]+num_features[4]+num_features[5]) + \
               [i+1]*(num_features[0]+num_features[1]+num_features[4]+num_features[5])

        col += list(range(0, num_features[0]+num_features[1]+num_features[4]+num_features[5])) + \
            list(range(0, num_features[0]+num_features[1]+num_features[4]+num_features[5]))
        
        data += hl_pos + positive_real_valued_human_features +\
                ml_pos + positive_real_valued_mouse_features +\
                hl_neg + negative_real_valued_human_features +\
                ml_neg + negative_real_valued_mouse_features

        i += 2 # read two examples in the while loop, one positive example and one negative example

    # Build a SciPy sparse matrix with feature data and convert it into an array
    X = scipy.sparse.coo_matrix((data,(row,col)),shape=(batch_size,num_features[0]+num_features[1]+num_features[4]+num_features[5])).toarray()
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
    def __init__(self,num_human_features,num_mouse_features,num_layers,num_neurons,dropout_rate):
        """
        num_layer_species: number of layers in species-specific sub-networks
        num_layer_final: number of layers in final sub-network
        num_neuron_human: number of neurons in each layer of human-specific sub-network
        num_neuron_mouse: number of neurons in each layer of mouse-specific sub-network
        num_neuron_final: number of neurons in each layer of final sub-network
        """
        super(PseudoSiameseNet, self).__init__()
        self.num_human_features = num_human_features
        self.num_layer_species = num_layers[0]
        self.num_layer_final = num_layers[2]
        self.num_neuron_human = num_neurons[:self.num_layer_species]
        self.num_neuron_mouse = num_neurons[2:2+self.num_layer_species]
        self.num_neuron_final = num_neurons[4:4+self.num_layer_final]

        # Sequence of operations done only on either human or mouse features
        self.human_layers = torch.nn.Sequential()
        self.mouse_layers = torch.nn.Sequential()
        for i in range(self.num_layer_species):
            if i==0: # from input features to first hidden layer
                self.human_layers.add_module('h'+str(i),
                                             torch.nn.Linear(num_human_features,int(self.num_neuron_human[i]),bias=False))
                self.mouse_layers.add_module('m'+str(i),
                                             torch.nn.Linear(num_mouse_features,int(self.num_neuron_mouse[i]),bias=False))
            else:
                self.human_layers.add_module('h'+str(i),
                                             torch.nn.Linear(int(self.num_neuron_human[i-1]),int(self.num_neuron_human[i])))
                self.mouse_layers.add_module('m'+str(i),
                                             torch.nn.Linear(int(self.num_neuron_mouse[i-1]),int(self.num_neuron_mouse[i])))
            self.human_layers.add_module('h'+str(i)+'dropout',torch.nn.Dropout(p=dropout_rate)) # dropout
            self.human_layers.add_module('hr'+str(i),torch.nn.ReLU()) # relu
            self.mouse_layers.add_module('m'+str(i)+'dropout',torch.nn.Dropout(p=dropout_rate)) # dropout
            self.mouse_layers.add_module('mr'+str(i),torch.nn.ReLU()) # relu

        # Sequence of operations done on concatenated output of species-specific sub-networks
        self.final_layers = torch.nn.Sequential()
        for i in range(self.num_layer_final):
            if i==0: # from concatenated output of species-specific sub-networks to the first layer of final sub-network
                self.final_layers.add_module('c'+str(i),
                                             torch.nn.Linear(int(self.num_neuron_human[-1])+int(self.num_neuron_mouse[-1]),int(self.num_neuron_final[i])))
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
        m = self.mouse_layers.forward(x[:,self.num_human_features:]) # mouse-specific sub-network
        c = torch.cat((h,m),1) # concatenate the output from human-specific sub-network and mouse-specific sub-network
        y = self.final_layers.forward(c) # final/final sub-network
        y = y.view(c.size()[0])
        return y
