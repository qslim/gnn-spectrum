import torch
from tqdm import tqdm


def filter_train_set(train_idx, dataset):
    num_filtered = 0
    filtered_train_idx = []
    train_idx = train_idx.numpy().tolist()
    for i in tqdm(range(len(train_idx)), desc="Filtering"):
        idx = train_idx[i]
        graph, _ = dataset[idx]
        if graph.num_edges() > 0:
            filtered_train_idx.append(idx)
        else:
            num_filtered = num_filtered + 1
    print("{} graphs with no edge filtered.".format(num_filtered))
    return torch.tensor(filtered_train_idx, dtype=torch.long)
