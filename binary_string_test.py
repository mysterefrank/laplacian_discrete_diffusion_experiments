import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def int_to_bits3(x):
    return np.array([(x>>i)&1 for i in range(3)][::-1], dtype=int)

def bits3_to_int(bits):
    x = 0
    for b in bits:
        x = (x<<1) | b
    return x

ALL_STATES = list(range(8))

def neighbors_3bit(x):
    neigh = []
    bits_x = int_to_bits3(x)
    for i in range(3):
        flipped = bits_x.copy()
        flipped[i] = 1 - flipped[i]
        y = bits3_to_int(flipped)
        neigh.append(y)
    return neigh

def laplacian_row(y):
    # L[y,y] = degree = 3
    row = [(y, 3)]
    for x in neighbors_3bit(y):
        row.append((x, -1))
    return row

# We'll define an *initial data distribution* p0 as uniform over 3 random states
np.random.seed(0)
chosen = np.random.choice(8, 3, replace=False)
p0 = np.zeros(8, dtype=float)
for c in chosen:
    p0[c] = 1.0
p0 /= p0.sum()
print("Using p0 uniform on states:", chosen)

"""
# Forward PDE in discrete steps *with sampling
#    pick an x0 from p0, then each step:
#      1. compute distribution for next step
#      2. sample from that distribution
#    This yields a trajectory x0 -> x1 -> ... -> xK
"""

def forward_euler_distribution(dist, dt=0.1):
    """
    dist: shape [8], distribution over states
    returns dist_next after one Euler step
    """
    dist_next = dist.copy()
    for y in ALL_STATES:
        row = laplacian_row(y)
        Lp = 0.0
        for (x, val) in row:
            Lp += val * dist[x]
        dist_next[y] += dt * (-Lp)
    # normalize
    total = dist_next.sum()
    if total>0:
        dist_next /= total
    return dist_next

def sample_from_dist(dist):
    """Given dist of shape [8], sample an integer state in [0..7]."""
    return np.random.choice(8, p=dist)

def sample_forward_trajectory(K=5, dt=0.1):
    """
    Start from p0, sample x0. Then for k in [0..K-1],
    do a forward Euler step of the distribution, and sample x_{k+1}.
    Return the states [x0, x1, ..., xK].
    """
    # distribution for k=0 is p0
    dist_k = p0.copy()
    x0 = sample_from_dist(dist_k)
    traj = [x0]

    # We'll keep track that x_k has distribution dist_k in the PDE sense
    for _ in range(K):
        # compute next distribution
        dist_kplus1 = forward_euler_distribution(dist_k, dt)
        # sample x_{k+1}
        x_next = sample_from_dist(dist_kplus1)
        traj.append(x_next)
        # the "next distribution" becomes dist_k for next iteration
        dist_k = dist_kplus1
    return traj

"""
Collect training data for the reverse model
Gather (x_{k+1}, k+1) -> x_{k}
"""

K = 5
num_trajs = 2000

pairs_input = []
pairs_output = []

for _ in range(num_trajs):
    traj = sample_forward_trajectory(K=K, dt=0.1)
    # traj is list of length (K+1)
    # For k in [0..K-1], we have x_{k+1}, want to predict x_k
    # We'll also store the integer time step (k+1).
    # Our input: (x_{k+1}, time=k+1), output: x_k
    for k in range(K):
        x_k = traj[k]
        x_kplus1 = traj[k+1]
        # time k+1 in [1..K]
        pairs_input.append((x_kplus1, k+1))
        pairs_output.append(x_k)

pairs_input = np.array(pairs_input, dtype=int)
pairs_output = np.array(pairs_output, dtype=int)
print(f"Collected {len(pairs_input)} reverse training pairs.")

"""
Define a small neural net that: 
    Input: x_{k+1} (one-hot) + time (scalar or embedding)
    Output: 8 logits for x_k
"""

class ReverseDenoiser(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        # We'll embed the input x_{k+1} as one-hot of dim=8
        # We'll embed the time step k+1 with a small embedding
        self.time_embed = nn.Embedding(K+1, 4)  # times in [1..K]
        # So total input = 8 (one-hot) + 4 (time embedding) = 12
        self.lin1 = nn.Linear(12, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin_out = nn.Linear(hidden_dim, 8)

    def forward(self, x_kplus1_onehot, t_idx):
        # x_kplus1_onehot: shape [B,8]
        # t_idx: shape [B], each in [1..K]
        t_emb = self.time_embed(t_idx)  # [B,4]
        x_in = torch.cat([x_kplus1_onehot, t_emb], dim=1)  # [B,12]
        h = F.relu(self.lin1(x_in))
        h = F.relu(self.lin2(h))
        logits = self.lin_out(h)  # shape [B,8]
        return logits

model = ReverseDenoiser()
optimizer = optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

# We'll convert the pairs_input and pairs_output to torch
pairs_input_torch = torch.from_numpy(pairs_input)   # shape [N,2], (x_{k+1}, time)
pairs_output_torch = torch.from_numpy(pairs_output) # shape [N], x_k

# We'll define a small dataset loader
dataset_size = len(pairs_input_torch)
batch_size = 64
inds = np.arange(dataset_size)
num_epochs = 10

model.train()
for epoch in range(num_epochs):
    np.random.shuffle(inds)
    total_loss = 0.0
    for start in range(0, dataset_size, batch_size):
        end = min(start+batch_size, dataset_size)
        batch_idx = inds[start:end]
        x_kplus1_vals = pairs_input_torch[batch_idx, 0]  # shape [B]
        t_vals = pairs_input_torch[batch_idx, 1]         # shape [B]
        x_k_vals = pairs_output_torch[batch_idx]         # shape [B]

        # One-hot encode x_{k+1}
        x_kplus1_onehot = F.one_hot(x_kplus1_vals, num_classes=8).float()

        logits = model(x_kplus1_onehot, t_vals)
        loss = loss_fn(logits, x_k_vals)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*(end-start)

    print(f"Epoch {epoch+1}, avg_loss={total_loss/dataset_size:.4f}")

"""
Use the learned model to sample from p_K -> p_{K-1} -> ... -> p_0
    We'll define p_K as uniform over states [0..7].
    Then at each step: x_{k}, ~ model( x_{k+1}, k+1 )
"""


def sample_reverse(model, K=5, num_samples=5):
    """
    Start from x_K ~ Uniform(0..7),
    then for k in [K..1], sample x_{k-1} via model.
    Return final x_0's or entire chain.
    """
    model.eval()
    chains = []
    for _ in range(num_samples):
        x_kplus1 = np.random.randint(8)  # from uniform
        chain = [x_kplus1]
        for step in reversed(range(1, K+1)):
            # step in [K..1], x_{step} -> x_{step-1}
            x_kplus1_onehot = F.one_hot(torch.tensor([x_kplus1]), 8).float()
            logits = model(x_kplus1_onehot, torch.tensor([step]))
            probs = F.softmax(logits, dim=1).detach().numpy()[0]  # shape [8]
            x_k = np.random.choice(8, p=probs)
            chain.append(x_k)
            x_kplus1 = x_k
        chain.reverse()  # so it goes x0..xK
        chains.append(chain)
    return chains

print("\nSampling from the learned reverse model (start from uniform at K=5):")
samples = sample_reverse(model, K=K, num_samples=8)
for s in samples:
    print("Chain:", s, " -> bits:", [int_to_bits3(x) for x in s])
