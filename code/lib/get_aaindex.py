import numpy as np
from sklearn import preprocessing

path = "./aaindex1.txt"


lines = open(path,"r").readlines()
new_lines = [line.strip() for line in lines]
print(len(new_lines))

'''
A/L     R/K     N/M     D/F     C/P     Q/S     E/T     G/W     H/Y     I/V

A R N D C Q E G H I L K M F P S T W Y V
ARNDCQEGHILKMFPSTWYV
'''

res = []

i = 0
while i<len(new_lines):
    line = new_lines[i].strip()
    if line.startswith("I "):
        x1 = new_lines[i]
        x2 = new_lines[i+1]
        x3 = new_lines[i+2]
        x2_tmp = x2.strip().split()
        x3_tmp = x3.strip().split()
        tmp = x2_tmp + x3_tmp

        # if 'NA' in tmp:
        #     pass
        #     print(tmp)
        # else:
        #     new_tmp = [float(x) for x in tmp]
        #     res.append(new_tmp)
        # i = i+3
        new_tmp = [float(x) if x != "NA" else 0 for x  in tmp ]
        res.append(new_tmp)
        i = i+3

    else:
        i = i +1

res = np.array(res).T

from sklearn.decomposition import PCA


res = preprocessing.minmax_scale(res)
print("res",res.shape)
# res = PCA(n_components=0.99).fit_transform(res)
# res = PCA(n_components=20).fit_transform(res)


print("res",res.shape)

f = open("aaindex1.my.csv","w")

std = "ARNDCQEGHILKMFPSTWYV"
for i in range(len(std)):
    tmp = std[i]+","+",".join([str(x) for x in list(res[i])]) +"\n"
    f.write(tmp)
f.close()


