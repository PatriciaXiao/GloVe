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

print("vocabulary saved")

vocab_size = len(out_df)
vocab_dim = 200

word2idx = dict(zip(data["word"], data["idx"]))

with open("political_vectors.txt") as f:
    vec = f.read().split("\n")


print(len(vec))
print(vocab_size)
print(len(in_df))

for v in vec[:-1]:
    v_info = v.split(" ")
    w = str(v_info[0])
    e = np.array([float(i) for i in v_info[1:] if len(i)])
    print(e)
    exit(0)

# np.savez_compressed(file_name, context=embedding_weights)