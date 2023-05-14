
"""
Occlusion sensitivity analysis for visualization of the neural networks

"""
import os
import numpy as np
import argparse


from utils.preprocessing import pca, gen_segment_data, preproc

from utils.featsextraction import trained_net, feature_extractor_rest

#from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pdb
# parser
parser = argparse.ArgumentParser(description='Visualization')

parser.add_argument('--data_path', type=str, default='/proj/zhuyong/camcan/cc700/meg/release004/BIDS_20190411/meg_rest_mf',
                    help="data path of trainning data")
parser.add_argument('--len_seg', type=float, default=10,
                    help="length of data segment in s")
parser.add_argument('--num_sub', type=int, default=10,
                    help="number of subjects to be load")
parser.add_argument('--path', type=str, default='results')

args = parser.parse_args()


# prepare data ================================================================
# =============================================================================

datapath = args.data_path
sub_idx = os.listdir(datapath)
sub_names = [idx + '_ses-rest_task-rest_proc-sss.fif' for idx in sub_idx[:args.num_sub]]

list_files = [os.path.join(datapath,idx,'ses-rest','meg',name) for idx, name in zip(sub_idx,sub_names)]

x_r, labels,_,_ = gen_segment_data(preproc(list_files, low_cutoff=.2, high_cutoff=45, notch=False),len_segment=args.len_seg, meg='grad')


# feature extractor============================================================
# =============================================================================



path = args.path  # 
    
model, model_parm = trained_net('./'+path, load_ema=False)

x, pca_parm = pca(x_r, params=model_parm['pca_parm']) # PCA

feats = feature_extractor_rest(x, model, model_parm)

# oclussion p channels=========================================================
# =============================================================================
n_ch = x.shape[0]

num_comp = model_parm['num_comp']
heatmap = np.zeros((n_ch,num_comp))
z0 = np.zeros(x.shape)

for comp in range(n_ch):
    # oclussion
    okla = np.zeros(x.shape)
    okla[:,:] = x[:,:]
    okla[comp,:] = z0[comp,:]

    # forward
    #x, pca_parm = pca(okla,num_comp=64, params=pca_parm) # PCA
    feateval_okla = feature_extractor_rest(okla, model, model_parm)

    corrmat = np.corrcoef(feateval_okla, feats,'Pearson')
    corrmat = corrmat[0:num_comp,num_comp:]
    heatmap[comp,:] = (np.diag(corrmat))

plt.figure(figsize=(20,16))
ax = sns.heatmap(heatmap)
plt.savefig('./heatmap-okla.png')

pdb.set_trace()
###############################################################################