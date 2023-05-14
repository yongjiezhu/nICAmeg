""" NICA(IIA) and NICA(TCL) model for MEG/EEG from multi-subjects """


import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



# =============================================================================
# NICA(TCL) model
# =============================================================================

class tcl(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_class,
                 num_layers, activation = 'Maxout', pool_size = 2,
                 slope = .1, feature_nonlinearity='abs'):
        """Build model with a feature-MLP and multi-MLR classfier for each subject
        Args:
            input_dim: (MLP) size of input channels, here is number of components
            hidden_dim: (MLP) size of nodes for each layer
            num_class: the dim of output (number of labels) with subject-specific
            num_layers: (MLP) number of layers
            num_sub: number of subjects (number of MLR)
            activation: (MLP) (option) activation function in the middle layer
            pool_size: (MLP) pool size of max-out nonlinearity
            slope: (MLP) for ReLU and leaky_relu activation
            feature_nonlinearity:(MLP) (option) Nonlinearity of the last hidden layer (feature value)
        """
        super(tcl, self).__init__()
        self.feature_nonlinearity = feature_nonlinearity
        
        # Shared feature-MLP
        self.MLP = MLP(input_dim, hidden_dim, num_layers,
                       activation = activation, pool_size = pool_size, slope = slope)
        
        # MLRs (subject-specific mlr)
        num_sub = len(num_class)
        if isinstance(hidden_dim, list):
            _mlr_input_dim = hidden_dim[-1]
        else:
            _mlr_input_dim = hidden_dim
            
        _MLRs_list = [nn.Linear(_mlr_input_dim, num_class[k]) for k in range(num_sub)]
        self.MLRs = nn.ModuleList(_MLRs_list)
        
        for k in range(num_sub):
            # initialize
            torch.nn.init.xavier_uniform_(self.MLRs[k].weight)

    
    def forward(self, x, sub_id=None):
        """forward pass
        Args:
            x: shape(batch_size,num_channels)
            sub_id: subject id
        Returns:
            y: labels (batch_size,)
            h: features (batch_size, num_channels)
        """
        h = self.MLP(x)
        
        if self.feature_nonlinearity == 'abs':
            h = torch.abs(h) # Nonlinearity of the last hidden layer (feature value)
            
        # logis
        if sub_id is None:
            y = None
        else:
            uniq = torch.unique(sub_id)
            y = [self.MLRs[k](h[(sub_id == k).nonzero().squeeze(),:]) for k in uniq]
            y = torch.concatenate(y,axis=0)
            
        return y, h

# =============================================================================
# NICA(IIA) model based on TCL
# =============================================================================
class itcl(nn.Module):
    def __init__(self, h_sizes, num_class, num_layers, h_sizes_z=None,
                 activation = 'Maxout', slope = .1, ar_order=1, pool_size=2):
        """ IIA network model for segment-wise stationary model
         Args:
             h_sizes: number of channels for each layer [num_layer+1] (first size is input-dim)
             num_class: number of classes with [num_sub,], here we assume each sub has the same number labels
             h_sizes_z: number of channels for each layer of MLP_z [num_layer+1] (first size is input-dim)
             ar_order: NVAR model order. e.g. 1st-order: x(t) = f(x(t-1),s(t)); 2rd: x(t) = f(x(t-1),x(t-2),s(t))
             pool_size: pool size of max-out nonlinearity
         """
        super(itcl, self).__init__()
        self.num_order = ar_order
        self.num_class = num_class
        
        ## shared MLP
        # h (feature extractor)
        self.layer = MLP(h_sizes[0], h_sizes[1:], num_layers,
                       activation = activation, pool_size = pool_size, slope = slope)
        
        # hz (auxiliary structure)
        if h_sizes_z is None:
            h_sizes_z = h_sizes.copy()
            h_sizes_z[0] = h_sizes[0] - np.int(h_sizes_z[0]/(self.num_order+1)) # exclude x(t)

        self.layerz = MLP(h_sizes_z[0], h_sizes_z[1:], num_layers,
                       activation = activation, pool_size = pool_size, slope = slope)
        
        # MLRs (subject-specific mlr)
        num_sub = len(num_class)
        _MLRs_list = [nn.Linear((h_sizes[-1] + h_sizes_z[-1])*2, num_class[k]) for k in range(num_sub)]
        self.MLRs = nn.ModuleList(_MLRs_list)
        for k in range(num_sub):
            # initialize
            torch.nn.init.xavier_uniform_(self.MLRs[k].weight)
            

    def forward(self, x, sub_id =None):
        """ forward
         Args:
             x: input [batch, dim]
             sub_id: subject id
         """
        batch_size, in_dim = x.size()
        num_comp = in_dim // (self.num_order+1)
        xz = x[:, num_comp:] # exclude x(t)

        # h
        h = self.layer(x)
        h_nonlin = torch.cat((h**2, h), 1)

        # hz
        hz = self.layerz(xz)
        hz_nonlin = torch.cat((hz**2, hz), 1)

        # concatenate
        hhz = torch.cat((h_nonlin, hz_nonlin), 1)
        # MLRs
        if sub_id is None:
            y = None
        else:
            uniq = torch.unique(sub_id)
            y = [self.MLRs[k](hhz[(sub_id == k).nonzero().squeeze(),:]) for k in uniq]
            y = torch.concatenate(y,axis=0)

        return y, h


# =============================================================================
# =============================================================================    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, 
                 num_layers, activation = 'Maxout', pool_size = 2,
                 slope = .1):
        """Built feature-MLP model as feature extractor:
        Args:
            input_dim: size of input channels, here is number of components
            hidden_dim: size of nodes for each layer
            num_layers: number of layers == len(hidden_dim)
            activation: (option) activation function in the middle layer
            pool_size: pool size of max-out nonlinearity
            slope: for ReLU and leaky_relu activation
        """
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        # Checking hidden_dim
        if isinstance(hidden_dim, int):
            self.hidden_dim = [hidden_dim] * (self.num_layers)
        elif isinstance(hidden_dim, list):
            self.hidden_dim = hidden_dim
        else:
            raise ValueError('Wrong argument type for hidden_dim: {}'.format(hidden_dim))
        
        # Activation
        if isinstance(activation, str):
            self.activation = [activation] * (self.num_layers - 1) # No activation in last layer
        elif isinstance(activation, list):
            self.activation = activation
        else:
            raise ValueError('Wrong argument type for activation: {}'.format(activation))
        
        self._act_f = []
        for act in self.activation:
            if act == 'lrelu':
                self._act_f.append(lambda x: F.leaky_relu(x, negative_slope=slope))
            elif act == 'ReLU':
                self._act_f.append(lambda x: F.relu(x))
            elif act == 'Maxout':
                self._act_f.append(Maxout(pool_size))
            elif act == 'sigmoid':
                self._act_f.append(F.sigmoid)
            elif act == 'none':
                self._act_f.append(lambda x: x)
            else:
                ValueError('Incorrect activation: {}'.format(act))

        # MLP
        if activation == 'Maxout':
            _layer_list = [nn.Linear(self.input_dim, self.hidden_dim[0]*pool_size)] # compensate maxout pool size
            for k in range(1,len(hidden_dim)-1):
                _layer_list.append(nn.Linear(self.hidden_dim[k - 1], self.hidden_dim[k]*pool_size)) # compensate maxout pool size
            _layer_list.append(nn.Linear(self.hidden_dim[-2], self.hidden_dim[-1]))
        else:
            _layer_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            for k in range(1,len(hidden_dim)):
                _layer_list.append(nn.Linear(self.hidden_dim[k - 1], self.hidden_dim[k]))
            
        self.layers = nn.ModuleList(_layer_list)
        
        # Dropout
        self.dropout = nn.Dropout(p=0.2)
        
        # Bacth-norm
        self.bn = nn.ModuleList()
        for bni in range(self.num_layers-1):
            self.bn.append(nn.BatchNorm1d(self.hidden_dim[bni]))
            
        # initialize
        for k in range(len(self.layers)):
            torch.nn.init.xavier_uniform_(self.layers[k].weight)
        
    def forward(self, x):
        """forward process
        Args:
            x: input data nput [batch, dim]
        Returns:
            h: features
        """
        #h/feature values
        h = x
        for k in range(len(self.layers)):
            if k == len(self.layers) - 1:
                h = self.layers[k](h)
            else:
                h = self._act_f[k](self.layers[k](h))
                if k<=1: h=self.dropout(h)

        return h
    
# =============================================================================
# =============================================================================
class Maxout(nn.Module):
    def __init__(self, pool_size):
        super().__init__()
        self._pool_size = pool_size

    def forward(self, x):
        m, _ = torch.max(torch.reshape(x, (*x.shape[:1], x.shape[1] // self._pool_size, self._pool_size, *x.shape[2:])), dim=2)
        return m


# =============================================================================
# =============================================================================
def sconcat_shifted_data(x, label=None, sub_ids=None, shift=1):
    """ Spatially concatenate temporally shifted signals to original ones for IIA model.
    Args:
        x: signals. 2D ndarray [num_comp, num_data]
        label: labels. 1D ndarray [num_data]
        shift: amount of temporal shift: [1,2,3]
    Returns:
        y: signals concatenated with their temporal shifts, y(t) = [x(t); x(t-1)]. 2D ndarray [2*num_comp, num_data-1]
        label: labels. 1D ndarray [num_data-1]
        sub_ids: subject id same size to labels
    """
    if not isinstance(shift, list):
        shift = [shift]

    xdim = x.shape[0]
    y = copy.copy(x)
    for sn in shift:
        if sn >= 1:
            xshift = np.concatenate([np.zeros([xdim,sn]), copy.copy(x[:,0:-sn])], axis=1)
        elif sn <=-1:
            xshift = np.concatenate([copy.copy(x[:,-sn:]), np.zeros([xdim,-sn])], axis=1)
        else:
            raise ValueError

        y = np.concatenate([y, xshift], axis=0)

    if np.max(shift) >= 1:
        y = y[:, np.max(shift):]
        if label is not None:
            label = label[0:-np.max(shift)]            
        if sub_ids is not None:
            sub_ids = sub_ids[0:-np.max(shift)]            
    if np.min(shift) <= -1:
        y = y[:, :np.min(shift)]
        if label is not None:
            label = label[:np.min(shift)]
        if sub_ids is not None:
            sub_ids = sub_ids[:np.min(shift)]

    return y, label, sub_ids

