# PasiBellettiStricescu2324

## Achieving Fairness without Demographics using Adversarially Reweighted Learning models

## Description
In much of the existing machine learning (ML) fairness research, pro- tected features like race and sex are typically included in datasets and used to address fairness concerns. However, due to privacy concerns and regulatory restrictions, collecting or using these features for training or in- ference is often not feasible. This raises the question: how can we train an ML model to be fair without knowing the protected group memberships? This work tackles this issue by proposing Adversarially Reweighted Learn- ing (ARL). The key idea is that non-protected features and task labels can help identify fairness problems. ARL uses these features to co-train an adversarial reweighting approach that enhances fairness. The results indicate that ARL improves Rawlsian Max-Min fairness and achieves sig- nificant AUC improvements for the worst-case protected groups across various datasets, outperforming current leading methods.

## Installation
Firstly, to run the code a Conda environment must be created and activated with the following commands:
```
conda env create -f environment.yml
```
Activate env
```
conda activate arl_uva 
```

## Usage
To print only the results of the project without further preprocessing run the file "results.ipynb".
To run the grid search for hyperparameter tuning, run the file "hyperparameters.py".

## Support
This code is based upon an existing re-implementation of a Google Research paper.
The original paper: Preethi Lahoti. 2020. Fairness without Demographics through Adversarially Reweighted Learning.
The re-implementation: J. Mohazzab, L.R. Weytingh, C.A. Wortmann and B. Brocades Zaalberg. 2021. Reim-
plementing the Adversarially Reweighted Learning model by Lahoti et al. (2020).

## Authors and acknowledgment
Reimplementation of the Google Paper by:
* J. Mohazzab
* L.R. Weytingh
* C.A. Wortmann
* B. Brocades Zaalberg

Further adaptation, development and code fixing and by: 
* A. Pasi
* C.R. Stricescu
* M. Belletti