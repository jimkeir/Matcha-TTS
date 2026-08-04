"""Length-bucketed batch sampler for the training dataloader.

Groups similar-length utterances into each batch so a batch pads only to its
own (near-uniform) longest member instead of the corpus long tail. This cuts
padding compute, bounds the padded-shape spread (torch.compile stays within
its recompile budget when combined with the collate's pad_quantum), and -
with a frame budget - bounds peak activation memory by construction.

Stochasticity: each epoch draws a fresh permutation seeded by (seed, epoch),
partitions it into pools of pool_factor * batch_size, sorts WITHIN each pool
by length, cuts batches, then shuffles the batch order. Expected gradients
are unchanged (padding is masked out of every loss); only per-step length
correlation is introduced - the standard trade every bucketing speech recipe
makes. Opt-in via data.bucket_batching; intended for real overnight training
runs (see Final_Training.md).

FORK-ONLY simplification (deliberate, 2026-08-04): single-device only - no
rank sharding. An upstream PR would need a distributed-aware rewrite.
"""

import numpy as np
import torch


class LengthBucketBatchSampler:
    def __init__(self, lengths, batch_size, pool_factor=50, frame_budget=None, seed=1234):
        if torch.distributed.is_available() and torch.distributed.is_initialized() \
                and torch.distributed.get_world_size() > 1:
            raise RuntimeError("LengthBucketBatchSampler is single-device only (fork-local simplification).")
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.pool = int(pool_factor) * self.batch_size
        self.frame_budget = int(frame_budget) if frame_budget else None
        self.seed = int(seed)
        self._epoch = 0

    def _schedule(self, epoch):
        rng = np.random.default_rng([self.seed, epoch])
        order = rng.permutation(len(self.lengths))

        batches = []
        for start in range(0, len(order), self.pool):
            chunk = order[start:start + self.pool]
            chunk = chunk[np.argsort(self.lengths[chunk], kind="stable")]
            i = 0
            while i < len(chunk):
                take = min(self.batch_size, len(chunk) - i)
                if self.frame_budget:
                    # Ascending order within the pool, so the running element is
                    # the batch max: cut when (count) * max_len would exceed the
                    # budget. Only the longest buckets ever split.
                    n = 0
                    while n < take and (n + 1) * self.lengths[chunk[i + n]] <= self.frame_budget:
                        n += 1
                    take = max(1, n)  # a single over-budget utterance still trains
                batches.append(chunk[i:i + take])
                i += take

        for j in rng.permutation(len(batches)):
            yield [int(x) for x in batches[j]]

    def __iter__(self):
        epoch = self._epoch
        self._epoch += 1
        return self._schedule(epoch)

    def __len__(self):
        # Peek at the upcoming epoch's schedule without consuming it. Exact,
        # at the cost of one extra schedule generation per epoch (cheap).
        return sum(1 for _ in self._schedule(self._epoch))
