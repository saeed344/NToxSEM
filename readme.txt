##Accurate identification of neurotoxic peptides and neurotoxins proteins based Stacked ensemble classifier with multi-perspective features


##dependencies:

Python 3.6
numpy
scipy
scikit-learn
pandas
Xgboost
SFS
GBM
##Guiding principles: **The dataset contains both training dataset and independent test set.

**feature extraction:
peptide_features.ipynb implement AAC,AAINDEX,BLOSUM62,DPC,PAAC and TPC
FEGS.ipynb implement FEGS
BERT.ipynb implement BERT
BioBERT
ESM2.ipynb implement ESM2-480 and ESM2-320
Fasttext.ipynb implement FastText
ProtTrans.ipynb implement ProtT5-BFD and ProtT5-UR50
RECM_CLBP.m implement RECM-CLBP
RECM-DCT.m implement RECM-DCT

** feature selection:

Lasso.ipynb implement Lasso.
XGB_SFS_selection.py implement XGB-SFS.
** Classifier:

Meta_EnsClassifiers.ipynb implements stacked ensemble classifier.

** Dataset:

combined folder contains the data of the training and independent of combined dataset.
protein folder contains the data of the training and independent of protein dataset.
peptide folder contains the data of the training and independent of peptide dataset.
