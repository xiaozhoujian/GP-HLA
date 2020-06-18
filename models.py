import torch.nn as nn
from utils import AA_IDX
import torch
from neural_tangents import stax
from jax import random


class Embedding(nn.Module):
    def __init__(self, embedded_dim, embedding_weights):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(AA_IDX) + 1, embedding_dim=embedded_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_weights, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

    def forward(self, x):
        x = self.embedding(x)
        return x


def wide_resnet_block(channels, strides=(1, 1), channel_mismatch=False):
    main = stax.serial(
        stax.Relu(), stax.Conv(channels, (3, 3), strides, padding='SAME'),
        stax.Relu(), stax.Conv(channels, (3, 3), padding='SAME')
    )
    shortcut = stax.Identity() if not channel_mismatch else stax.Conv(channels, (3, 3), strides, padding='SAME')
    return stax.serial(stax.FanOut(2),
                       stax.parallel(main, shortcut),
                       stax.FanInSum())


def wide_resnet_group(n, channels, strides=(1, 1)):
    blocks = []
    blocks += [wide_resnet_block(channels, strides, channel_mismatch=True)]
    for _ in range(n - 1):
        blocks += [wide_resnet_block(channels, (1, 1))]
    return stax.serial(*blocks)


def wide_resnet(block_size, k, num_classes):
    return stax.serial(
        stax.Conv(16, (3, 3), padding='SAME'),
        wide_resnet_group(block_size, int(16 * k)),
        wide_resnet_group(block_size, int(32 * k), (2, 2)),
        wide_resnet_group(block_size, int(64 * k), (2, 2)),
        stax.AvgPool((8, 8)),
        stax.Flatten(),
        stax.Dense(num_classes, 1., 0.))

