import numpy as np 
import pandas as pd 

in_df = pd.read_csv("political_vocab.txt", sep=" ", header=None)

data = {
    "idx": list(range(len(in_df)+1)), 
    "word": ["<UNK>"] + [str(w).replace('\x01', ' ') for w in list(in_df[0])],
    "count": [0] + list(in_df[1])}
# print(data["count"])

out_df = pd.DataFrame(data=data)
out_df.to_csv("vocab.csv", sep="\t", index=None)