# Unsupervised representation learning of spontaneous MEG data with nonlinear ICA
This repository contains code to show how to train nonlinear ICA with resting-state MEG and then be used to dowenstream classification tasks on label-limited data sets, which is presented in [Unsupervised representation learning of spontaneous MEG data with nonlinear ICA](https://www.sciencedirect.com/science/article/pii/S1053811923002938?via%3Dihub), published at NeuroImage 2023.
Work done by **Yongjie Zhu** (U. of Helsinki), **Tiina Parviainen** (U. of Jyväskylä), **Erkka Heinilä** (U. of Jyväskylä), **Lauri Parkkonen** (Aalto) and **Aapo Hyvärinen** (U. of Helsinki).

## Dependencies

This project was tested with the following versions:

- python 3.8
- PyTorch 1.13.1
- mne 1.3.0
- scikit-learn 1.2.1
- pickle 4.0

## Usage
The well-trained models are saved under ‎`results/`. You can download it and use it to your specific data (e.g., for classification). You can also train the model with your data by type ‎`python nICA_trainning.py --data_path "with your path"` to train the NICA model with your data.  
e.g. trainning NICA(TCL) model with CamCan:
```
python nICA_trainning.py --data_path /camcan/meg_rest -m tcl
```
Type ‎`python nICA_trainning.py --help` to learn about hyperparameters.

For the downstream tasks, type ‎`python nICA_downstreams.py --data_path "with your path"` to perform specific downstream tasks with trained NICA models. 
e.g.:
```
python nICA_downstreams.py --data_path /camcan/meg_passive --path results/
```

Refer to `utils/vis.py` for visualizing the neural networks with occlusion sensitivity analysis. You can easily modify some parameters in ‎`utils/preprocessing.py` with your styles to preprocess the MEG data. ‎`utils/featsextraction.py` contains functions for feature extraction using trained models. The bash job can also be used to run multiple experiments in parallel (e.g. with different random seeds) on a SLURM-equipped server.

## Reference

If you find this code helpful/inspiring for your research, we would be grateful if you cite the following:

```bib
@article{zhu2023unsupervised,
  title={Unsupervised representation learning of spontaneous MEG data with Nonlinear ICA},
  author={Zhu, Yongjie and Parviainen, Tiina and Heinilä, Erkka and Parkkonen, Lauri and Hyvärinen, Aapo},
  journal={NeuroImage},
  volume = {274},
  pages={120142},
  year={2023},
  publisher={Elsevier}
}
```
```bib
@inproceedings{hyvarinen2016unsupervised,
  title={Unsupervised feature extraction by time-contrastive learning and nonlinear ica},
  author={Hyvärinen, Aapo and Morioka, Hiroshi},
  booktitle={Advances in Neural Information Processing Systems (NIPS2016)},
  volume={29},
  address={Barcelona, Spain},
  year={2016}
}
```
```bib
@inproceedings{morioka2021independent,
  title={Independent innovation analysis for nonlinear vector autoregressive process},
  author={Morioka, Hiroshi and Hälvä, Hermanni and Hyvärinen, Aapo},
  booktitle={Proc.\ Artificial Intelligence and Statistics (AISTATS2021)},
  pages={1549--1557},
  year={2021},
  organization={PMLR}
}
```

## License
A full copy of the license can be found [here](LICENSE).
