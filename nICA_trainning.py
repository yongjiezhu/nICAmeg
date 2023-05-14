"""
NICA model trainning wit resting-state data, i.e., pretext tasks.

"""
import os
import argparse
from utils.preprocessing import pca, preproc, gen_segment_data

import pickle
import shutil

from model.nica import sconcat_shifted_data
from model import nica

#==============================================================================
# parsing argments

def parse():
    """Parser of argments for all paras
    """
    parser = argparse.ArgumentParser(description='NICA-CamCam training args:')
    
    # CamCan data args
    parser.add_argument('--data_path', type=str, default='/proj/camcan/meg_rest_mf',
                        help="data path of trainning data")
    parser.add_argument('--len_seg', type=float, default=5,
                        help="length of data segment in s")
    parser.add_argument('--meg', type=str, default='grad',
                        help="mne pick types")
    parser.add_argument('--num_sub', type=int, default=2,
                        help='number of subject to be load')
    parser.add_argument('--pca', type=float, default=True,
                        help='pca preprocessing')
    
    # model args
    parser.add_argument('-m', type=str, default='itcl',
                        help="model (tcl or itcl)")
    parser.add_argument('--num_comp', type=int, default=20,
                        help="number of components extracted by h")
    
    # NVAR moder oder
    parser.add_argument('--ar_order', type=int, default=2,
                        help="number of model order for IIA")
    parser.add_argument('--shift_order', type=int, default=2,
                        help="number of shift order e.g. if =3, regard x(t-3) as x(t-1)")
    
    # MLP & training
    parser.add_argument('--act', type=str, default='ReLU',
                        help='activation of MLP')
    parser.add_argument('--num_layer', type=int, default=3,
                        help='number of mlp hiddin layer')
    parser.add_argument('--rand_seed', type=int, default=0,
                        help='training seed')
    parser.add_argument('--init_lr', type=float, default=.01,
                        help='initial learning rate')
    parser.add_argument('--mom', type=float, default=.9,
                        help='momentum parameter of SGD')
    parser.add_argument('--max_steps', type=int, default=int(3e1),
                        help='number of iterations')
    parser.add_argument('--decay_steps', type=int, default=int(1e1),
                        help='decay steps')
    parser.add_argument('--decay_factor', type=float, default=.1,
                        help='decay factor')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='mini-batch size')
    parser.add_argument('--m_ave_decay', type=float, default=.999,
                        help='moving average decay of variables to be saved')
    parser.add_argument('--checkpoint_steps',type=int, default=1e7,
                        help='interval to save checkpoint')
    parser.add_argument('--summary_steps', type=int, default=10000,
                        help='interval to save summary')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='weight decay')
    
    # saving
    parser.add_argument('--out_dir', type=str, default='results')
    
    args = parser.parse_args()
    return args

args = parse()

#==============================================================================
# MEG/EEG data preprocessing (add your data path and the file naming rules accordingly)
#==============================================================================
#data path includs all subjects' resting-state MEG .fif data
datapath = args.data_path 
subs = os.listdir(datapath)
sub_names = [idx + '_ses-rest_task-rest_proc-sss.fif' for idx in subs[:args.num_sub]]

list_files = [os.path.join(datapath,idx,'ses-rest','meg',name) for idx, name in zip(subs,sub_names)]

# generate labels  with segment-wise stationarity for pretext classification 
# (preproc() for sensor level, use preproc_source() for source level)
x_rest, labels, subids, num_class = gen_segment_data(preproc(list_files, low_cutoff=.2, high_cutoff=45, notch=False),\
                                                     len_segment=args.len_seg, meg=args.meg)

# Preprocessing -----------------------------------------------
pca_parm = None
if args.pca: x_rest, pca_parm = pca(x_rest,num_comp=64) # PCA if ture

###############################################################################
# define model (NICA(IIA) or NICA(TCL))
###############################################################################

list_hidden_nodes = [args.num_comp*(2**i) for i in reversed(range(args.num_layer-1))] + [args.num_comp]
#list_hidden_nodes = [30]*(args.num_layer-1) + [args.num_comp]

list_hidden_nodes_z = None

if args.m == 'itcl': # NICA(IIA)
    # concatenate temporally shifted data
    x, y, z = sconcat_shifted_data(x_rest, labels, subids, shift=[(i+1)*args.shift_order for i in range(args.ar_order)]) 
    model = nica.itcl([x.shape[0]]+list_hidden_nodes,
                      num_class=num_class, activation=args.act, ar_order=args.ar_order, num_layers=args.num_layer)
    
elif args.m == 'tcl':  # NICA(TCL)
    x, y, z = x_rest, labels, subids
    model = nica.tcl(input_dim=x.shape[0], hidden_dim=list_hidden_nodes,
                      num_class=num_class, activation=args.act, num_layers=args.num_layer)
else:
    raise ValueError('Wrong argument type for model param')

model_parm = {'random_seed':args.rand_seed,
          'num_comp':args.num_comp,
          'num_layer':args.num_layer,
          'list_hidden_nodes':list_hidden_nodes,
          'list_hidden_nodes_z':list_hidden_nodes_z,
          'moving_average_decay':args.m_ave_decay,
          'pca_parm':pca_parm,
          'num_class':num_class if 'num_class' in locals() else None,
          'ar_order':args.ar_order,
          'shift_order':args.shift_order,
          'activation':args.act,
          'model':args.m}
#==============================================================================
# PyTorch dataloader
#==============================================================================
from utils.dataloader import SegmentDataset, train_test_split, get_weights

import torch
from torch.utils.data import SequentialSampler  
from torch.utils.data import WeightedRandomSampler  
from torch.utils.data import DataLoader 

device = 'cuda'
dataset = SegmentDataset(x.T, y, z)
ds_train, ds_valid = train_test_split(dataset, n_splits=args.max_steps)

weights_train = get_weights(ds_train)
weights_train = torch.as_tensor(weights_train).to(device)

batch_size_train = args.batch_size
batch_size_valid = args.batch_size
# sampler_train = RandomSampler(ds_train)

sampler_train = WeightedRandomSampler(weights_train, len(ds_train))
sampler_valid = SequentialSampler(ds_valid)

# create loaders
num_workers = 0
loader_train = \
    DataLoader(ds_train, batch_size=batch_size_train,
               num_workers=num_workers, sampler=sampler_train)
loader_valid = \
    DataLoader(ds_valid, batch_size=batch_size_valid,
               num_workers=num_workers, sampler=sampler_valid)

#==============================================================================
# TCL training
#==============================================================================
from model.train import train  
import torch.optim as optim
from utils.featsextraction import feature_extractor_rest
from sklearn.decomposition import FastICA

model.to(device)
optimizer = optim.SGD(model.parameters(), lr=args.init_lr, momentum=args.mom, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_steps, gamma=args.decay_factor)

best_model, train_loss, val_loss = train(model, loader_train, loader_valid, optimizer, scheduler, args.max_steps, device)

# further linear ICA for NICA(TCL)
W_ = None
if args.m == 'tcl':
    feats = feature_extractor_rest(x, best_model, model_parm)
    ica = FastICA()
    feats = ica.fit_transform(feats.T)
    W_ = ica.components_ # linear unmixing matrix
model_parm['W'] = W_   

#==============================================================================
# saving
#==============================================================================
# # Note: save folder must be under 
train_dir_base = './' + args.out_dir

train_dir = os.path.join(train_dir_base, 'trainedNN') # save directory (Caution!! this folder will be removed at first)

saveparmpath = os.path.join(train_dir, 'parm.pkl') # file name to save parameters


# =============================================================
# =============================================================

# Prepare save folder -----------------------------------------
if train_dir.find("/"+args.out_dir+"/") > -1:
    if os.path.exists(train_dir):
        print("delete savefolder: {0:s}...".format(train_dir))
        shutil.rmtree(train_dir)  # Remove folder
    print("make savefolder: {0:s}...".format(train_dir))
    os.makedirs(train_dir)  # Make folder
else:
    assert False, "savefolder looks wrong"

# Save trained model ----------------------------------
save_path = os.path.join(train_dir, 'model.pt')
print("Save model in file: {0:s}".format(save_path))
torch.save({'model_state_dict': best_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()}, save_path)
    
# Save parameters necessary for evaluation --------------------


print("Save parameters...")
with open(saveparmpath, 'wb') as f:
    pickle.dump(model_parm, f, pickle.DEFAULT_PROTOCOL)

print('Done')