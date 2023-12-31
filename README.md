# AIPBoost project
## AIPBoost: Identifying Anti-inflammatory Peptide by Boost Learning Based on Two-Stage Curated Feature and Model from Peptide Sequence Information   
Identifying anti-inflammatory peptides (AIPs) is crucial for treating inflammatory conditions. However, the traditional experimental approach is time-consuming and costly. In this paper, we propose AIPBoost to accurately identify AIPs. Our work presents two key contributions. Firstly, a two-stage feature filter strategy is proposed to identify 40 informative meta-features from top-8 group encoding features including Charge, Hydrophobicity, AAindex1, PaDEL, Prot-T5, CTF, AAC and ESM-2 that show better performance to identify AIPs compared with 18 group encoding methods. Secondly, a model importance measurement is designed by using the predicting probability of candidate models to find suitable meta-models. Based on the informative meta-features and meta-models, a two-layer ensemble model is designed. The first layer integrates random forest and extremely randomized trees to learn the difference of AIPs and non-AIPs, and the second layer employs logistic regression to predict AIPs. The comparison with eight state-of-the-art models on two benchmark datasets demonstrate superior performance with improvements of MCC (5.5% to 11.68%) and MCC (6.66% to 31.3%) on AIP2125 and AIP4194 test dataset. By utilizing a two-stage feature filter and a two-layer ensemble learning model, AIPBoost achieves higher accuracy in identifying AIPs. Our work contributes to finding informative features and an efficient framework to identify AIPs.

# Preparation  
create a python=3.8 environment  
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  
pip install numpy  
pip install biopython  
pip install scikit-learn  
pip install rdkit  
pip install padelpy  
pip install transformers  

# How to use AIPBoost?  
in the code/AIPBoost.py file, please replace the "seq" and run python AIPBoost.py The result will show in result.csv file. 
