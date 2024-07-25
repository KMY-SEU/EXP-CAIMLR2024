import numpy as np
import torch


def generate_trajectory(configs):
    rads = np.arange(0, configs.seq_len) * (2 * np.pi / (configs.seq_len - 1))
    trajs = np.vstack([np.cos(rads), np.sin(rads)])

    for d in range(2, configs.enc_in):
        trajs = np.vstack([trajs, trajs[-1] * np.sin(rads)])
        trajs[-2] = trajs[-2] * np.cos(rads)

    return torch.tensor((trajs.T)[np.newaxis, :, :], dtype=torch.float)


def compute_trajectory_length(trajs):
    _, T, N = trajs.shape

    dist = 0.
    for t in range(T - 1):
        dist += np.linalg.norm(trajs[:, t + 1] - trajs[:, t])

    return dist


def compute_normalized_squared_length(trajs):
    _, T, N = trajs.shape

    dist = 0.
    for t in range(T):
        dist += np.sum(np.square(trajs[:, t]))

    return dist / (N * T)


def compute_angular_entropy(trajs1, trajs2):
    B, T, N = trajs1.shape

    theta = []
    for b in range(B):
        for t in range(T):
            dot = trajs1[b, t].dot(trajs2[b, t])
            norms = np.linalg.norm(trajs1[b, t]) * np.linalg.norm(trajs2[b, t])

            theta += [np.arccos(dot / norms)]

    probs, _ = np.histogram(theta, bins=50, density=True)

    entropy = 0.
    for i, p in enumerate(probs):

        if i < len(probs) - 1:
            entropy += (-p * np.log(p) * (_[i + 1] - _[i]))
        else:
            entropy += (-p * np.log(p) * (np.pi - _[i]))

    return entropy
