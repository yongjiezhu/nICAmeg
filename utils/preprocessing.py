""" MEG data preprocessing with MNE"""


import os
import os.path as op
import numpy as np
import mne
from mne.preprocessing import ICA
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs

# =============================================================================
# =============================================================================
def scale(X):
    """Standard scaling of data
    Args:
        X: array, shape (num_channels,num_times)
    Returns:
        Xt: sacaled data, array with shape (num_channels,num_times)
    """
    if X.ndim == 2:
        X -= np.mean(X, axis=1)[:, None]
        Xt = X / np.std(X, axis=1)[:, None]
    elif X.ndim == 3:
        X -= np.mean(X, axis=2)[:,:, None]
        Xt = X / np.std(X, axis=2)[:,:, None]
    else:
        assert False, "ndim erro in scaleing"
    
    return Xt

# =============================================================================
# =============================================================================
def scale4task(X, pre_sti=40):
    """Standard scaling of data based on pre-stimulus baseline
    Args:
        X: array, shape (num_epoch, num_channels,num_times)
        pre_sti: time points pre onset
    Returns:
        Xt: sacaled data, array with shape (num_epoch, num_channels,num_times)
    """
    x0 = X[:,:,:pre_sti]
    X -=np.mean(x0, axis=2)[:,:, None]
    return X #/ np.std(x0, axis=2)[:,:, None]

# =============================================================================
# =============================================================================
def load_task_rawdata(list_files, low_cutoff=0.2, high_cutoff=None, notch=True, down_fs=200):
    """load and preproc task-session data
    Args:
        list_files: list of raw.fif path
        low_cutoff: low frequency cutoff
        high_cutoff: high frequency cutoff
        notvh: do notvh filter
        down_fs: down sampling frequency
        
    Returns:
        concatedData: array, shape (num_epochs,num_channels,num_times)
        labels: array, shape (num_epochs,)
    """
    print('Preprocessing....')
    
    list_data = []
    list_label = []
    for fname in list_files:
        # in case error happen when mne.io.reading
        try: 
            raw = mne.io.read_raw_fif(fname, preload=True)
            
            if notch: raw.notch_filter(freqs=(50, 100, 150, 200))
            raw.resample(sfreq=down_fs)
            raw.filter(l_freq=low_cutoff, h_freq=high_cutoff)
            # ICA
            ica = ICA(method='fastica',n_components=0.99,max_iter=1000)
            picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False,stim=False, exclude='bads')
            ica.fit(raw, picks=picks)
            eog_idx,eog_scores=ica.find_bads_eog(raw,threshold=0.55,measure='correlation')
            ecg_idx,ecg_scores=ica.find_bads_ecg(raw)
            ica.apply(raw,exclude=eog_idx+ecg_idx)
            # ica done
            events = mne.find_events(raw,stim_channel='STI101', min_duration=0.003,output='onset')
            events = mne.merge_events(events,[6,7,8],10)
            picks = mne.pick_types(raw.info,meg='grad')
            
            epochs = mne.epochs.Epochs(raw,events,tmin=-.3,tmax=.5,detrend=1,reject={'grad':4000e-13},picks=picks)
            epochs.equalize_event_counts(['9','10'])
            del raw
            
            data = epochs.get_data()

            labels = epochs.events[:,2]-9
            del epochs
            
            list_data.append(data)
            list_label.append(labels)
            
        except IOError:
            continue
        except ValueError:
            continue
        except KeyError:
            continue
        
    #return np.concatenate(list_data,axis=0), np.concatenate(list_label,axis=0)
    return list_data, list_label

# =============================================================================
# sensor level analysis
# =============================================================================
def preproc(list_files, low_cutoff=0.2, high_cutoff=None, notch=True, down_fs=200):
    """Preprocessing: 1, remove low-freq drifts
                      2, notch filter 50, 100... Hz
                      3, resample to 200 Hz
    Args:
        list_files: list of raw.fif path
        low_cutoff: low frequency cutoff
        high_cutoff: high frequency cutoff
        notvh: do notvh filter
        down_fs: down sampling frequency
        
    Returns:
        list_proc: list of preprocessed raw MNE data
    """
    print('Preprocessing....')
    list_proc = []
    for fname in list_files:
        try:
            raw = mne.io.read_raw_fif(fname, preload=True)
            raw.filter(l_freq=low_cutoff, h_freq=high_cutoff)
            if notch:
                raw.notch_filter(freqs=(50, 100, 150, 200))
                
            raw.resample(sfreq=down_fs)
            #ica reject EOG ECG
            ica = ICA(method='fastica',n_components=0.99,max_iter=1000)
            picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False,stim=False, exclude='bads')
            ica.fit(raw, picks=picks)
            eog_idx,eog_scores=ica.find_bads_eog(raw,threshold=0.55,measure='correlation')
            ecg_idx,ecg_scores=ica.find_bads_ecg(raw)
            ica.apply(raw,exclude=eog_idx+ecg_idx)
            
            list_proc.append(raw)
        except IOError:
            continue
        except ValueError:
            continue
        except KeyError:
            continue
    
    return list_proc
# =============================================================================
# =============================================================================
def gen_segment_data(list_proc, len_segment, meg=True):
    """Generate segment data, the labels and subjects id for each data point
    Args:
        list_proc: list of preprocessed raw MNE data
        len_segment: lengh of segment in seconds
        meg: mne.pick_types para， If True include MEG channels. 
            If string it can be ‘mag’, ‘grad’, ‘planar1’ or ‘planar2’ to select only magnetometers, all gradiometers
    Returns:
        concatedData: arraty data with (num_channels,num_times) from all subs
        labels: labels (num_times,)
        subject ID: sub index which teach data point come from
        num_class: number of class for each subjects (for subject-specific MLR)
    """
    print('Generating labels...')
    list_data = []
    list_lable = []
    list_subid = []
    list_num_class = []
    sfreq = list_proc[0].info['sfreq']
    num_samples_seg = int(sfreq*len_segment) # number of data-points in each segment
    
    for n, raw in enumerate(list_proc):
        timepoints = sfreq*9*60 # raw.n_times = 9min, we assume the same duration for each subjects
        n_segments = int(timepoints/num_samples_seg) #int(raw.n_times/num_samples_seg) # number of segments for each sub !!!!!!!!!!
        meg_channels = mne.pick_types(raw.info, meg=meg, eeg=False, eog=False)
        data = raw.get_data(picks=meg_channels)
        #data = scale(data)
        
        list_data.append(data[:,:n_segments*num_samples_seg])
        
        #labels
        labels = np.zeros(num_samples_seg*n_segments) # for each subject
        for si in range(n_segments):
            start_idx = num_samples_seg*si
            end_idx = num_samples_seg*(si+1)
            labels[start_idx:end_idx] = si
        list_lable.append(labels)
        
        # subid
        list_subid.append(n*np.ones(num_samples_seg*n_segments))
        
        # number of classes
        list_num_class.append(n_segments)
    
    return np.concatenate(list_data,axis=1), np.concatenate(list_lable,axis=0), np.concatenate(list_subid,axis=0), list_num_class


# =============================================================================
# prepare resting-state source level data
# =============================================================================

def preproc_source(len_segment,subs,src_methd='MNE', mode='mean', doica=True, low_cutoff=0.2, high_cutoff=None, notch=True, down_fs=200):
    """
    """
    datadir = '/proj/camcan-bids/derivatives/'
    preprodir = '/proj/camcan/'
    meg_dir = op.join(datadir,'maxfilter')   # MEG_dir
    subjects_dir = op.join(datadir,'recon')  # MRI_data_path
    
    # read fsaverage src and 
    fname_src_fs = os.path.join(subjects_dir,'fsaverage','bem','fsaverage-ico4-src.fif')
    src_fs = mne.read_source_spaces(fname_src_fs,patch_stats=True)
    labels = mne.read_labels_from_annot('fsaverage', parc='aparc_sub', subjects_dir=subjects_dir) #aparc.a2009s aparc_sub
    
    sub_idx = os.listdir(meg_dir)
    tcs = []

    list_subid = [] # for trainning
    list_num_class = []
    n_flag = int(0)
    for subid in sub_idx[:subs]:
        try:
            fname_raw = op.join(meg_dir,subid,'meg',subid + '_task-rest_proc-tsss_meg.fif')
            fname_cov = op.join(preprodir,'emptyroom',subid[4:],'empty_room_tsss_cov.fif')
            fname_trans = op.join(preprodir,'processed','trans_spm','trans-new','trans-fid-new',
                           subid + '-new-fid-AR-trans.fif')
            fname_bem = op.join(preprodir,'processed', 'cc700','mri','pipeline',
                          'release004','BIDSsep','megraw',subid,'meg',
                          subid + '-5120-5120-5120-bem-sol.fif')
            fname_src = op.join(preprodir,'processed','cc700','mri','pipeline',
                          'release004','BIDSsep','megraw',subid,'meg',
                          subid + '-ico4-src.fif')
            # read raw MEG, preprocess and epochs
            raw = mne.io.read_raw_fif(fname_raw, preload=True)
                
            if notch: raw.notch_filter(freqs=(50, 100, 150, 200))
            raw.resample(sfreq=down_fs)
            raw.filter(l_freq=low_cutoff, h_freq=high_cutoff)
            
            # ica reject EOG ECG
            if doica:
                ica = ICA(method='fastica',n_components=0.99,max_iter=1000)
                picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False,stim=False, exclude='bads')
                ica.fit(raw, picks=picks)
                eog_idx,eog_scores=ica.find_bads_eog(raw,threshold=0.55,measure='correlation')
                ecg_idx,ecg_scores=ica.find_bads_ecg(raw)
                ica.apply(raw,exclude=eog_idx+ecg_idx)
            #ica done
            
            events = mne.make_fixed_length_events(raw, duration=len_segment)
            epochs = mne.Epochs(raw, events=events, tmin=0, tmax=len_segment,
                                baseline=None, reject={'grad':4000e-13}, preload=True)
            #epochs.apply_hilbert(envelope=True)  # faster to apply in sensor space
            # read cov and src
            cov = mne.read_cov(fname_cov)
            src = mne.read_source_spaces(fname_src,patch_stats=True)
            # fwd
            fwd = mne.make_forward_solution(epochs.info, trans=fname_trans, src=src, bem=fname_bem,
                                            meg=True, eeg=False, mindist=5.0, n_jobs=2)
            # inv
            inverse_operator = make_inverse_operator(epochs.info, fwd, cov, loose=0.2,depth=0.8)
            method = src_methd # dSPM MNE
            snr = 1. # Standard assumption for average data but using it for single trial
            lambda2 = 1. / snr ** 2
            
            stc = apply_inverse_epochs(epochs, inverse_operator, lambda2, method, label=None, pick_ori="normal")
            
            # morph to fsavearge src
            stc_fs = [mne.compute_source_morph(stc_t, subid, src_to=src_fs, subjects_dir=subjects_dir,
                                              smooth=5, verbose='error').apply(stc_t) for stc_t in stc]
            
            #labels = mne.read_labels_from_annot(subid, parc='aparc', subjects_dir=subjects_dir)
            
            label_ts = mne.extract_label_time_course(stc_fs, labels, src_fs, mode=mode,
                                             return_generator=False)
            tcs = tcs + label_ts
            
            # generate labels, id
            num_samples_seg = label_ts[0].shape[1]
            list_num_class.append(len(label_ts))
            
            list_subid.append(n_flag*np.ones(num_samples_seg*len(label_ts)))
            list_lable = [np.zeros(label_ts[0].shape[1]) + k for k in range(len(label_ts))]
            
        except IOError:
            continue
        except ValueError:
            continue
        except KeyError:
            continue
        except FileNotFoundError:
            continue
        except RuntimeError:
            continue
    
    return np.concatenate(tcs,axis=1), np.concatenate(list_lable,axis=0), np.concatenate(list_subid,axis=0), list_num_class
        
# =============================================================================
# prepare task source level data
# =============================================================================
def preparetask_source(src_methd='MNE', mode='mean', doica=True,low_cutoff=0.2, high_cutoff=None, notch=True, down_fs=200):
    """
    """
    datadir = '/proj/camcan-bids/derivatives/'
    preprodir = '/proj/camcan/'
    meg_dir = '/scratch/work/passive'   # MEG_dir
    subjects_dir = op.join(datadir,'recon')  # MRI_data_path
    
    # read fsaverage src and 
    fname_src_fs = os.path.join(subjects_dir,'fsaverage','bem','fsaverage-ico4-src.fif')
    src_fs = mne.read_source_spaces(fname_src_fs,patch_stats=True)
    labels = mne.read_labels_from_annot('fsaverage', parc='aparc_sub', subjects_dir=subjects_dir)
    
    sub_idx = os.listdir(meg_dir)
    list_tcs = []
    list_sti = []
    for subid in sub_idx:
        try:
            fname_raw = op.join(meg_dir,subid,'ses-passive','meg',subid + '_ses-passive_task-passive_proc-sss.fif')
            fname_cov = op.join(preprodir,'emptyroom',subid[4:],'empty_room_tsss_cov.fif')
            fname_trans = op.join(preprodir,'processed','trans_spm','trans-new','trans-fid-new',
                           subid + '-new-fid-AR-trans.fif')
            fname_bem = op.join(preprodir,'processed', 'cc700','mri','pipeline',
                          'release004','BIDSsep','megraw',subid,'meg',
                          subid + '-5120-5120-5120-bem-sol.fif')
            fname_src = op.join(preprodir,'processed','cc700','mri','pipeline',
                          'release004','BIDSsep','megraw',subid,'meg',
                          subid + '-ico4-src.fif')
            # read raw MEG, preprocess and epochs
            raw = mne.io.read_raw_fif(fname_raw, preload=True)
                
            if notch: raw.notch_filter(freqs=(50, 100, 150, 200))
            raw.resample(sfreq=down_fs)
            raw.filter(l_freq=low_cutoff, h_freq=high_cutoff)

            # ica reject EOG ECG
            if doica:
                ica = ICA(method='fastica',n_components=0.99,max_iter=1000)
                picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False,stim=False, exclude='bads')
                ica.fit(raw, picks=picks)
                eog_idx,eog_scores=ica.find_bads_eog(raw,threshold=0.55,measure='correlation')
                ecg_idx,ecg_scores=ica.find_bads_ecg(raw)
                ica.apply(raw,exclude=eog_idx+ecg_idx)
            #ica done
            
            events = mne.find_events(raw,stim_channel='STI101', min_duration=0.003,output='onset')
            events = mne.merge_events(events,[6,7,8],10)
            epochs = mne.epochs.Epochs(raw,events,tmin=-.3,tmax=.5,detrend=1,reject={'grad':4000e-13})
            epochs.equalize_event_counts(['9','10'])
            
            # read cov and src
            cov = mne.read_cov(fname_cov)
            src = mne.read_source_spaces(fname_src,patch_stats=True)
            # fwd
            fwd = mne.make_forward_solution(epochs.info, trans=fname_trans, src=src, bem=fname_bem,
                                            meg=True, eeg=False, mindist=5.0, n_jobs=2)
            # inv
            inverse_operator = make_inverse_operator(epochs.info, fwd, cov, loose=0.2,depth=0.8)
            method = src_methd # dSPM MNE
            snr = 1. # Standard assumption for average data but using it for single trial
            lambda2 = 1. / snr ** 2
            
            stc = apply_inverse_epochs(epochs, inverse_operator, lambda2, method, label=None, pick_ori="normal")
            
            # morph to fsavearge src
            stc_fs = [mne.compute_source_morph(stc_t, subid, src_to=src_fs, subjects_dir=subjects_dir,
                                              smooth=5, verbose='error').apply(stc_t) for stc_t in stc]
            
            #labels = mne.read_labels_from_annot(subid, parc='aparc', subjects_dir=subjects_dir)
            
            label_ts = mne.extract_label_time_course(stc_fs, labels, src_fs, mode=mode,
                                             return_generator=False)
            tcs = np.array(label_ts)
            list_tcs.append(tcs)
            stis = epochs.events[:,2]-9
            list_sti.append(stis)
        except IOError:
            continue
        except ValueError:
            continue
        except KeyError:
            continue
        except FileNotFoundError:
            continue
        except RuntimeError:
            continue
    
    return list_tcs, list_sti
# ============================================================
# from Dr. Hiroshi Morioka
# ============================================================
def pca(x, num_comp=None, params=None, zerotolerance = 1e-7):
    """Apply PCA whitening to data.
    Args:
        x: data. 2D ndarray [num_comp, num_data]
        num_comp: number of components
        params: (option) dictionary of PCA parameters {'mean':?, 'W':?, 'A':?}. If given, apply this to the data
        zerotolerance: (option)
    Returns:
        x: whitened data
        parms: parameters of PCA
            mean: subtracted mean
            W: whitening matrix
            A: mixing matrix
    """
    print("PCA...")

    # Dimension
    if num_comp is None:
        num_comp = x.shape[0]
    print("    num_comp={0:d}".format(num_comp))

    # From learned parameters --------------------------------
    if params is not None:
        # Use previously-trained model
        print("    use learned value")
        data_pca = x - params['mean']
        x = np.dot(params['W'], data_pca)

    # Learn from data ----------------------------------------
    else:
        # Zero mean
        xmean = np.mean(x, 1).reshape([-1, 1])
        x = x - xmean

        # Eigenvalue decomposition
        xcov = np.cov(x)
        d, V = np.linalg.eigh(xcov)  # Ascending order
        # Convert to descending order
        d = d[::-1]
        V = V[:, ::-1]

        zeroeigval = np.sum((d[:num_comp] / d[0]) < zerotolerance)
        if zeroeigval > 0: # Do not allow zero eigenval
            raise ValueError

        # Calculate contribution ratio
        contratio = np.sum(d[:num_comp]) / np.sum(d)
        print("    contribution ratio={0:f}".format(contratio))

        # Construct whitening and dewhitening matrices
        dsqrt = np.sqrt(d[:num_comp])
        dsqrtinv = 1 / dsqrt
        V = V[:, :num_comp]
        # Whitening
        W = np.dot(np.diag(dsqrtinv), V.transpose())  # whitening matrix
        A = np.dot(V, np.diag(dsqrt))  # de-whitening matrix
        x = np.dot(W, x)

        params = {'mean': xmean, 'W': W, 'A': A}

        # Check
        datacov = np.cov(x)

    return x, params


