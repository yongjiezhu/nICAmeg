"""
Downstream tasks (classification) using well-trained NICA models 

"""

import os
import argparse
import numpy as np
from utils.preprocessing import pca, scale4task, load_task_rawdata

from utils.featsextraction import trained_net, feature_extractor

from sklearn.model_selection import LeaveOneOut
#from sklearn.metrics import accuracy_score
from sklearn import svm
#==============================================================================
# parsing argments

def parse():
    """Parser of argments for all paras
    """
    parser = argparse.ArgumentParser(description='NICA-CamCam downstream task args:')
    
    parser.add_argument('--data_path', type=str, default='/proj/camcan/meg_passive_mf',
                        help="data path of downstream task")    

    parser.add_argument('-w', type=float, default=.05,
                        help='length of sliding window across epoch')
    parser.add_argument('--stride', type=float, default=.025,
                        help='stride of sliding window across epoch')
    parser.add_argument('--num_sub', type=int, default=5,
                        help="number of subjects to be load")
    parser.add_argument('--device', type=str, default='cpu',
                        help='training on device')
    parser.add_argument('--pca', type=float, default=True,
                        help='pca preprocessing')
    parser.add_argument('--path', type=str, default='results',
                        help="well trained model path")
    
    args = parser.parse_args()
    return args

args = parse()

#==============================================================================
# load trained model and model parameters
#==============================================================================
model, model_parm = trained_net('./'+args.path, load_ema=False)

#==============================================================================
# MEG/EEG preprocessing (add your data path and the file naming rules accordingly)
#==============================================================================
# datapath includs all subjects' task MEG .fif files
datapath = args.data_path
sub_idx = os.listdir(datapath)

sub_names = [idx + '_ses-passive_task-passive_proc-sss.fif' for idx in sub_idx]

list_files = [os.path.join(datapath,idx,'ses-passive','meg',name) \
              for idx, name in zip(sub_idx,sub_names)]

data_list, labels_list = load_task_rawdata(list_files, low_cutoff=.2, high_cutoff=45, notch=False)

## for leave one out
subid = []
for idx in range(len(data_list)):
    n_epo = data_list[idx].shape[0]
    subid.append(np.ones([n_epo,],dtype=int)*idx)
subid = np.concatenate(subid)
##
# catenated across subjects
data, epoch_label = np.concatenate(data_list,axis=0), np.concatenate(labels_list,axis=0)

# if pca
if args.pca:
    n_com = model_parm['pca_parm']['W'].shape[0]
    n_ep, n_t = data.shape[0], data.shape[2]
    data = np.transpose(data, [1,2,0])
    data = np.reshape(data,[data.shape[0],data.shape[2]*data.shape[1]], order='F') # matlab-like
    data, _ = pca(data, num_comp=n_com)
    data = np.reshape(data,[n_com,n_t,n_ep],order='F')
    data = np.transpose(data, [2,0,1])
    
#==============================================================================
# feature extraction and downstream classification
#==============================================================================

feats = scale4task(feature_extractor(data, model, model_parm))[:,:,60:] ### 80=0.4*200 presti

# sliding windows for event-related data=======================================
# =============================================================================

sfreq, n_times = 200, feats.shape[2]
n_wpts, n_stride = int(sfreq*args.w), int(sfreq*args.stride)
n_winds = int((n_times - n_wpts) / n_stride) + 1

Xn = []
for i in range(n_winds):
    Xi = np.mean(feats[:,:,i*n_stride:i*n_stride+n_wpts], axis=2)
    Xn.append(Xi)

X = np.concatenate(Xn, axis=1)
y = epoch_label

# SVM classification===========================================================
# =============================================================================

loo = LeaveOneOut().split(range(len(data_list)))
for idout, (train, test) in enumerate(loo):
    
    X_train, y_train = X[subid!=test[0],:], y[subid!=test[0]]
    X_test, y_test = X[subid==test[0],:], y[subid==test[0]]
    
    clf = svm.LinearSVC(penalty='l2', dual=False, C=2, max_iter=2000)
    clf.fit(X_train, y_train)
    sc = clf.score(X_test, y_test)
    print('Test acu:%.4f' % sc)