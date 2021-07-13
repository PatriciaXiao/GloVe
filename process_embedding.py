import numpy as np 
import pandas as pd 

in_df = pd.read_csv("political_vocab.txt", sep=" ", header=None)


data = {
    "idx": list(range(len(in_df)+1)), 
    "word": ["<UNK>"] + [str(w).replace('\x01', ' ') for w in list(in_df[0])],
    "count": [0] + list(in_df[1])}
# print(data["count"])

out_df = pd.DataFrame(data=data)
out_df.to_csv("vocab_v1.csv", sep="\t", index=None)

print("vocabulary ver1 saved")


vocab_size = len(out_df)
vocab_dim = 200

word2idx = dict(zip(data["word"], data["idx"]))

"""
print(len(vec))
print(vocab_size)
print(len(in_df))

print(set([l.split(" ")[0] for l in vec[:-1]]) - set(data["word"]))
exit(0)
"""

data_out = {
    "idx": [0],
    "word": ["<UNK>"]
}

embedding_weights = np.zeros((1, 200))

with open("political_vectors.txt") as f:
    vec = f.read().split("\n")

for v in vec[:-1]:
    v_info = v.split(" ")
    w = str(v_info[0]).replace('\x01', ' ')
    if len(w) == 0: continue
    e = np.array([float(i) for i in v_info[1:] if len(i)]).reshape(1,200)
    embedding_weights = np.concatenate((embedding_weights, e), axis=0)
    print(embedding_weights.shape)
    exit(0)

# np.savez_compressed(file_name, context=embedding_weights)