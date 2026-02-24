## NToxSEM
NToxSEM: Accurate identification of neurotoxic peptides and neurotoxin proteins based on a stacked ensemble classifier with multi-perspective features
dependencies:
Python 3.6
numpy
scipy
scikit-learn
pandas
Xgboost
SFS
GBM

## feature extraction:
peptide_features.ipynb implement AAC,AAINDEX,BLOSUM62,DPC,PAAC and TPC
FEGS.ipynb implements FEGS
BERT.ipynb implements BERT
BioBERT
ESM2.ipynb implement ESM2-480 and ESM2-320
Fasttext.ipynb implements FastText
ProtTrans.ipynb implement ProtT5-BFD and ProtT5-UR50
RECM_CLBP.m implements RECM-CLBP
RECM-DCT.m implements RECM-DCT
## feature selection:
Lasso.ipynb implements Lasso.
XGB_SFS_selection.py implements XGB-SFS.
Classifier:
Meta_EnsClassifiers.ipynb implements a stacked ensemble classifier.
Dataset:
##Guiding principles: **The dataset contains both a training dataset and an independent test set.
The combined folder contains the training and independent datasets.
The protein folder contains the data of the training and the independent protein dataset.
The peptide folder contains the data for the training and independent peptide datasets.
