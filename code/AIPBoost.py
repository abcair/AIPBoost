'''
https://github.com/scikit-learn-contrib/boruta_py
'''

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from boruta import BorutaPy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from Bio import SeqIO
import pickle
from rdkit import Chem
from padelpy import from_smiles

index = np.load("./index_support.npy").astype(np.int32)

def read_fa(path):
    rx = SeqIO.parse(path,format="fasta")
    res = {}
    for x in rx:
        id = str(x.id)
        seq = str(x.seq).upper().replace("B","")
        res[id] = seq
    return res

def Charge(seq):
    '''
    G	5.97
    A	6
    S	5.68
    P	6.3
    V	5.96
    T	6.16
    C	5.07
    I	6.02
    L	5.98
    N	5.41
    D	2.77
    Q	5.65
    K	9.74
    E	3.22
    M	5.74
    H	7.59
    F	5.48
    R	10.76
    Y	5.66
    W	5.89

    '''
    std_charge={

    "G":	5.97 ,
    "A":	6    ,
    "S":	5.68 ,
    "P":	6.3  ,
    "V":	5.96 ,
    "T":	6.16 ,
    "C":	5.07 ,
    "I":	6.02 ,
    "L":	5.98 ,
    "N":	5.41 ,
    "D":	2.77 ,
    "Q":	5.65 ,
    "K":	9.74 ,
    "E":	3.22 ,
    "M":	5.74 ,
    "H":	7.59 ,
    "F":	5.48 ,
    "R":	10.76,
    "Y":	5.66 ,
    "W":	5.89 ,

    }

    seq = seq[0:50]
    res = np.zeros(50)
    for i,x in enumerate(seq):
        res[i] = std_charge[x]
    return list(res)

def Hydrophobicity(seq):
    '''
    I	4.5
    V	4.2
    L	3.8
    F	2.8
    C	2.5
    M	1.9
    A	1.8
    G	-0.4
    T	-0.7
    S	-0.8
    W	-0.9
    Y	-1.3
    P	-1.6
    H	-3.2
    E	-3.5
    Q	-3.5
    D	-3.5
    N	-3.5
    K	-3.9
    R	-4.5
    '''
    std_Hyd={
    "I":	"4.5  ",
    "V":	"4.2  ",
    "L":	"3.8  ",
    "F":	"2.8  ",
    "C":	"2.5  ",
    "M":	"1.9  ",
    "A":	"1.8  ",
    "G":	"-0.4",
    "T":	"-0.7",
    "S":	"-0.8",
    "W":	"-0.9",
    "Y":	"-1.3",
    "P":	"-1.6",
    "H":	"-3.2",
    "E":	"-3.5",
    "Q":	"-3.5",
    "D":	"-3.5",
    "N":	"-3.5",
    "K":	"-3.9",
    "R":	"-4.5",
    }

    seq = seq[0:50]
    res = np.zeros(50)
    for i,x in enumerate(seq):
        res[i] = std_Hyd[x]
    return list(res)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

std = ["A", "I", "L", "V", "F", "W", "Y", "N", "C", "Q", "M", "S", "T", "R", "H", "K", "D", "E", "G", "P"]
std_dict = {}
for x in std:
    smiles = Chem.MolToSmiles(Chem.MolFromFASTA(x))
    descriptors = from_smiles(smiles)
    res = [float(x) if is_number(x) else 0 for x in list(descriptors.values())]
    std_dict[x] = np.array(res)

def get_padel(seq):
    seq = seq[0:50]
    res = 0
    for x in seq:
        res = res + std_dict[x]
    return list(res)

def BPF(seq):
    std = ["A", "I", "L", "V", "F", "W", "Y", "N", "C", "Q", "M", "S", "T", "R", "H", "K", "D", "E", "G", "P"]
    seq = seq[0:10]
    res = []
    for x in seq:
        tmp = [0]*20
        tmp[std.index(x)] = 1
        res.extend(tmp)
    while len(res)<=400:
        res.append(0)
    return res

def AAC(seq):
    std = ["A", "I", "L", "V", "F", "W", "Y", "N", "C", "Q", "M", "S", "T", "R", "H", "K", "D", "E", "G", "P"]
    res = []
    for x in std:
        tmp = seq.count(x) / len(seq)
        res.append(tmp)
    return res

path = "./lib/aaindex1.my.csv"
lines = open(path).readlines()
aaindex1 = {}
for line in lines:
    tmps = line.strip().split(",")
    id = tmps[0]
    vec = [float(x) for x in tmps[1:]]
    aaindex1[id] = vec

def get_aaindex1(seq):
    seq = seq[0:50]
    res = 0
    for i, x in enumerate(seq):
        res = res + np.array(aaindex1[x])
    return list(res)

from Bio import SeqIO
import torch
from multiprocessing import cpu_count
torch.set_num_threads(cpu_count())
import numpy as np

model_esm, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
model_esm = model_esm

def infer_esm(seq):
    batch_converter = alphabet.get_batch_converter()
    model_esm.eval()  # disables dropout for deterministic results
    # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    data = [
        ("tmp", seq),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    batch_tokens = batch_tokens
    # batch_tokens = batch_tokens

    with torch.no_grad():
        results = model_esm(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]
    token_representations = token_representations.detach().cpu().numpy()
    # print(token_representations.shape)
    token_representations = token_representations[0][1:-1,:].sum(axis=0)
    # (7, 1280)
    return list(token_representations)

import re
import torch
from Bio import SeqIO
from transformers import T5Tokenizer, T5Model,T5EncoderModel
tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50',do_lower_case=False,)
model_t5 = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
model_t5 = model_t5

def infer_t5(seq):
    sequences_Example = [" ".join(list(seq))]
    sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]
    ids = tokenizer.batch_encode_plus(sequences_Example,
                                      add_special_tokens=True,
                                      padding=False)

    input_ids = torch.tensor(ids['input_ids'])
    attention_mask = torch.tensor(ids['attention_mask'])

    input_ids = input_ids
    attention_mask = attention_mask

    with torch.no_grad():
        # embedding = model(input_ids=input_ids,attention_mask=attention_mask,decoder_input_ids=None)
        embedding = model_t5(input_ids=input_ids,
                          attention_mask=attention_mask,
                          # decoder_input_ids=input_ids,
                          )

    # For feature extraction we recommend to use the encoder embedding
    encoder_embedding = embedding.last_hidden_state[0,:-1].detach().cpu().numpy().sum(axis=0)
    return list(encoder_embedding)


def infer(seq):
    x1 = Charge(seq)
    x2 = Hydrophobicity(seq)
    x3 = get_padel(seq)
    x4 = BPF(seq)
    x5 = AAC(seq)
    x6 = get_aaindex1(seq)
    res = x1 + x2 + x3 + x4 + x5 + x6
    return res


def predict(seq):
    data_fs = infer(seq)
    data_esm = infer_esm(seq)
    data_t5 = infer_t5(seq)
    data = np.array(data_fs + data_esm + data_t5)
    data = data[np.newaxis,:]
    data = data[:,index]
    et_path = open("./model/et.pkl","rb")
    rf_path = open("./model/rf.pkl","rb")
    lr_path = open("./model/lr.pkl","rb")

    et = pickle.load(et_path)
    rf = pickle.load(rf_path)
    lr = pickle.load(lr_path)

    data_et = rf.predict_proba(data)
    data_rf = et.predict_proba(data)

    layer2_data = np.concatenate([data_et,data_rf],axis=1)

    pred_res = lr.predict_proba(layer2_data)[:,1]

    flag = ""
    if pred_res[0]>0.5:
        flag = "AIP"
    else:
        flag= "non-AIP"

    f = open("result.csv","w")
    tmp = "seq,score,label"
    f.write(tmp+"\n")
    f.write(seq + "," + str(pred_res[0]) + "," + flag + "\n")
    f.close()

seq = "GIINTLQKYYCRVRGGRCAVLSCLPKEEQIGKCSTRGRKCCRRKK"
print(seq)
predict(seq)

