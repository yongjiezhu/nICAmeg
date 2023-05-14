"""
feature extraction from trained NICA models

"""

import os
import numpy as np
import shutil
import tarfile

import torch
from model import nica
from model.nica import sconcat_shifted_data
import pickle

from utils.preprocessing import scale4task
from sklearn.model_selection import LeaveOneOut
from sklearn import svm


def valid_by_svm(model, device, model_parm, data, labels, svm_parm):
    
    feats = scale4task(feature_extractor(data, model, model_parm, device))[:,:,60:] ### 80=0.4*200 presti
    
    # sliding windows==============================================================
    # =============================================================================
    
    sfreq, n_times = 200, feats.shape[2]
    n_wpts, n_stride = int(sfreq*svm_parm['w']), int(sfreq*svm_parm['stride'])
    n_winds = int((n_times - n_wpts) / n_stride) + 1
    
    Xn = []
    for i in range(n_winds):
        Xi = np.mean(feats[:,:,i*n_stride:i*n_stride+n_wpts], axis=2)
        Xn.append(Xi)
    
    X = np.concatenate(Xn, axis=1)
    y = labels
    
    # SVM classification===========================================================
    # =============================================================================
    subid = svm_parm['subid']
    cnt, msc = 0, 0
    for train, test in LeaveOneOut().split(range(svm_parm['num_sub'])):
        
        X_train, y_train = X[subid!=test[0],:], y[subid!=test[0]]
        X_test, y_test = X[subid==test[0],:], y[subid==test[0]]
        
        clf = svm.LinearSVC(penalty='l2', dual=False, C=2, max_iter=2000)
        clf.fit(X_train, y_train)
        sc = clf.score(X_test, y_test)
        #print('Test acu:%.4f' % sc)
        msc +=sc
        cnt +=1
    
    return msc/cnt

# =============================================================================
# read trained NICA model and model params
# =============================================================================
def trained_net(path, device ='cpu', load_ema = True):
    """Extract features from trained IIA net
    Args:
        path: str, path of trained model net
    Returns:
        model: trained net
        model_parm: model parameters
    """
    # parse model path
    eval_dir = os.path.join(path, 'trainedNN')
    parmpath = os.path.join(eval_dir, 'parm.pkl')
    
    if eval_dir.find('.tar.gz') >= 0:
        unzipfolder = "./storage/temp_temp"
        unzip(eval_dir, unzipfolder)
        eval_dir = unzipfolder
        parmpath = os.path.join(unzipfolder, 'parm.pkl')
    
    modelpath = os.path.join(eval_dir, 'model.pt')
    
    # Load parameter file
    with open(parmpath, 'rb') as f:
        model_parm = pickle.load(f)
    
    list_hidden_nodes = model_parm['list_hidden_nodes']
    list_hidden_nodes_z = model_parm['list_hidden_nodes_z'] if 'list_hidden_nodes_z' in model_parm else None
    num_segment = model_parm['num_class'] if 'num_class' in model_parm else None
    
    ar_order = model_parm['ar_order']
    num_channels = model_parm['pca_parm']['W'].shape[0]
    # define model
    if model_parm['model'] == 'itcl':
        dim_input_h = num_channels*(ar_order+1)
        dim_input_hz = num_channels*ar_order
        model = nica.itcl([dim_input_h] + list_hidden_nodes,
                          h_sizes_z=[dim_input_hz] + list_hidden_nodes_z if list_hidden_nodes_z is not None else None,
                          num_class=num_segment, activation=model_parm['activation'], num_layers=model_parm['num_layer'],
                          ar_order=ar_order)
    else:
        model = nica.tcl(input_dim=num_channels, hidden_dim=list_hidden_nodes,
                         num_class=num_segment, activation=model_parm['activation'], num_layers=model_parm['num_layer'])
    
    model = model.to(device)
    
    # load paras
    checkpoint = torch.load(modelpath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if load_ema: model.load_state_dict(checkpoint['ema_state_dict'])
    
    return model, model_parm

# =============================================================================
# extract features from epoch data using trained model
# =============================================================================
def feature_extractor(data, model, model_parm, device='cpu'):
    """Extract features from task (epoch) data using trained nica models
    Args:
        data: array, shape (num_epochs,num_channels,num_times)
        model: well-trained net
        model_parm: saved model parms
    Returns:
        feat: array, shape (num_epochs,num_comp,num_times)
    """
    model.to(device)
    ar_order = model_parm['ar_order']
    if 'shift_order' in model_parm:
        shift_order = model_parm['shift_order']
        
    else:
        shift_order = 1

    shift = [(i+1)*shift_order for i in range(ar_order)]
    num_comp = model_parm['num_comp']
    num_epochs = data.shape[0]
    
    if model_parm['model'] == 'itcl':
        features = np.zeros((num_epochs, num_comp, data.shape[2]-np.max(shift)))
    else:
        features = np.zeros((num_epochs, num_comp, data.shape[2]))
        
    model.eval()
    for epi in range(num_epochs):
        x = data[epi,:,:]
        if model_parm['model'] == 'itcl':
            x, _, _ = sconcat_shifted_data(x, shift=shift)
        x = x.T
        
        #forward
        x_torch = torch.from_numpy(x.astype(np.float32)).to(device)
        logits, h = model(x_torch)
        
        h_val = np.squeeze(h.detach().cpu().numpy())
        
        features[epi,:,:] = h_val.T
    
    return features

# =============================================================================
# extract features from resting data using trained model
# =============================================================================
def feature_extractor_rest(data, model, model_parm, device='cpu'):
    """Extract features from resting-state data using trained nica models
    Args:
        data: array, shape (num_channels,num_times)
        model: well-trained net
        model_parm: saved model parms
    Returns:
        feat: array, shape (num_comp,num_times)
    """
    model.to(device)
    ar_order = model_parm['ar_order']
    if 'shift_order' in model_parm:
        shift_order = model_parm['shift_order']
        
    else:
        shift_order = 1

    shift = [(i+1)*shift_order for i in range(ar_order)]
        
    model.eval()
    
    x = data.copy()
    if model_parm['model'] == 'itcl':
        x, _, _ = sconcat_shifted_data(x, shift=shift)
    x = x.T
    
    #forward
    x_torch = torch.from_numpy(x.astype(np.float32)).to(device)
    logits, h = model(x_torch)
    
    h_val = np.squeeze(h.detach().cpu().numpy())
    
    return h_val.T


# =============================================================================
# =============================================================================
def unzip(loadfile, unzipfolder, necessary_word='/storage'):
    """
    unzip trained model (loadfile) to unzipfolder
    """

    print('load: %s...' % loadfile)
    if loadfile.find(".tar.gz") > -1:
        if unzipfolder.find(necessary_word) > -1:
            if os.path.exists(unzipfolder):
                print('delete savefolder: %s...' % unzipfolder)
                shutil.rmtree(unzipfolder)  # remove folder
            archive = tarfile.open(loadfile)
            archive.extractall(unzipfolder)
            archive.close()
        else:
            assert False, "unzip folder doesn't include necessary word"
    else:
        if os.path.exists(unzipfolder):
            print('delete savefolder: %s...' % unzipfolder)
            shutil.rmtree(unzipfolder)  # remove folder
        os.makedirs(unzipfolder)
        src_files = os.listdir(loadfile)
        for fn in src_files:
            full_file_name = os.path.join(loadfile, fn)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, unzipfolder + '/')
        # unzipfolder = loadfile

    if not os.path.exists(unzipfolder):
        raise ValueError