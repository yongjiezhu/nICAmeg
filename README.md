# Unsupervised representation learning of spontaneous MEG data with nonlinear ICA
This repository contains code to show how to train nonlinear ICA with resting-state MEG and then be used to dowenstream classification tasks on smaller data sets, which is presented in [Unsupervised representation learning of spontaneous MEG data with nonlinear ICA](https://www.sciencedirect.com/science/article/pii/S1053811923002938?via%3Dihub), published at NeuroImage 2023.
Work done by **Yongjie Zhu** (U. of Helsinki), **Tiina Parviainen** (U. of Jyväskylä), **Erkka Heinilä** (U. of Jyväskylä), **Lauri Parkkonen** (Aalto) and **Aapo Hyvärinen** (U. of Helsinki).

## Dependencies

This project was tested with the following versions:

- python 3.8
- PyTorch 1.13.1
- mne 1.3.0
- scikit-learn 1.2.1
- pickle 0.7.5

## Usage
Type ‎`python nICA_trainning.py --data_path "with your path"` to train the NICA model with your data. Type ‎`python nICA_trainning.py -h` to learn about hyperparameters. The well-trained models are saved under ‎`results/`. Type ‎`python nICA_downstreams.py --data_path "with your path"` to perform specific downstream tasks with trained NICA models. Refer to `utils/vis.py` for visualizing the neural networks with occlusion sensitivity analysis. You can easily modify some parameters in ‎`utils/preprocessing.py` with your styles to preprocess the MEG data. 

## Reference

If you find this code helpful/inspiring for your research, we would be grateful if you cite the following:

```bib
@article{zhu2023unsupervised,
  title={Unsupervised representation learning of spontaneous MEG data with Nonlinear ICA},
  author={Zhu, Yongjie and Parviainen, Tiina and Heinil{\"a}, Erkka and Parkkonen, Lauri and Hyv{\"a}rinen, Aapo},
  journal={NeuroImage},
  volume = {274},
  pages={120142},
  year={2023},
  publisher={Elsevier}
}
```
```bib
@article{hyvarinen2016unsupervised,
  title={Unsupervised feature extraction by time-contrastive learning and nonlinear ica},
  author={Hyv{\"a}rinen, Aapo and Morioka, Hiroshi},
  journal={Advances in neural information processing systems},
  volume={29},
  year={2016}
}
```
```bib
@inproceedings{morioka2021independent,
  title={Independent innovation analysis for nonlinear vector autoregressive process},
  author={Morioka, Hiroshi and H{\"a}lv{\"a}, Hermanni and Hyvarinen, Aapo},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={1549--1557},
  year={2021},
  organization={PMLR}
}
```

## License
A full copy of the license can be found [here](LICENSE).
