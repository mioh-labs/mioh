"""Small signed mutex graph reference and non-CELF object feature memory.

Deterministic research CPU decoder, not an optimized production implementation.
No centres, fixed instance count, shape cuts or detector-box merges.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .o3 import OFFSETS


def mutex_partition(support, affinities, uncertainty_band=.05):
    support = np.asarray(support, bool)
    affinities = np.asarray(affinities, np.float32)
    height, width = support.shape
    if affinities.shape != (len(OFFSETS), height, width) or not np.isfinite(affinities).all():
        raise ValueError("invalid affinity map")
    parent = np.arange(height*width)
    size = np.ones(height*width, np.int32)
    mutex = [set() for _ in range(height*width)]

    def root(cell):
        while parent[cell] != cell:
            parent[cell] = parent[parent[cell]]
            cell = parent[cell]
        return int(cell)

    sources, destinations, scores, attractive = [], [], [], []
    grid = np.arange(height*width).reshape(height, width)
    for channel, (dy, dx) in enumerate(OFFSETS):
        y0, y1 = max(0, -dy), min(height, height-dy)
        x0, x1 = max(0, -dx), min(width, width-dx)
        if y0 >= y1 or x0 >= x1:
            continue
        a = (slice(y0, y1), slice(x0, x1))
        b = (slice(y0+dy, y1+dy), slice(x0+dx, x1+dx))
        probability = affinities[channel][a]
        valid = support[a] & support[b] & (abs(probability-.5) > uncertainty_band)
        sources.extend(grid[a][valid].tolist())
        destinations.extend(grid[b][valid].tolist())
        scores.extend(abs(2*probability[valid]-1).tolist())
        attractive.extend((probability[valid] > .5).tolist())
    order = np.argsort(-np.asarray(scores), kind="stable")
    for edge in order:
        a, b = root(sources[edge]), root(destinations[edge])
        if a == b:
            continue
        if not attractive[edge]:
            mutex[a].add(b); mutex[b].add(a)
            continue
        if b in mutex[a]:
            continue
        if size[a] < size[b]:
            a, b = b, a
        neighbors = (mutex[a] | mutex[b]) - {a, b}
        for neighbor in neighbors:
            mutex[neighbor].discard(a); mutex[neighbor].discard(b)
            mutex[neighbor].add(a)
        mutex[a] = neighbors
        mutex[b].clear()
        parent[b] = a
        size[a] += size[b]
    labels = np.zeros(height*width, np.int32)
    identities = {}
    for cell in np.flatnonzero(support):
        representative = root(int(cell))
        identity = identities.setdefault(representative, len(identities)+1)
        labels[cell] = identity
    return labels.reshape(height, width)


@dataclass
class MemoryEntry:
    prototype: np.ndarray
    mask: np.ndarray
    last: int
    velocity: tuple[float, float] = (0, 0)


class ObjectMemory:
    """A bounded bank read by the network. Only own predicted masks enter it.

    Matching is provisional IoU/feature Hungarian matching, not a proof of ID.
    A missing or ambiguous match holds its old prototype rather than overwriting it.
    """
    def __init__(self, capacity=32, ttl=32):
        self.capacity, self.ttl = capacity, ttl
        self.entries = {}
        self.next_id = 1

    def features(self, dimension=16):
        return np.stack([entry.prototype for entry in self.entries.values()]) if self.entries else np.zeros((0, dimension), np.float32)

    def update(self, frame, labels, embedding, confidence):
        self.entries = {identity: entry for identity, entry in self.entries.items() if frame-entry.last <= self.ttl}
        regions = []
        for value in np.unique(labels):
            mask = labels == value
            if value == 0 or mask.sum() < 4 or confidence[mask].mean() < .6:
                continue
            prototype = embedding[:, mask].mean(axis=1)
            prototype /= max(1e-8, np.linalg.norm(prototype))
            regions.append((mask, prototype))
        old_ids = sorted(self.entries)
        cost = np.ones((len(regions), len(old_ids)), np.float32)*2
        for a, (mask, prototype) in enumerate(regions):
            for b, identity in enumerate(old_ids):
                entry = self.entries[identity]
                dy, dx = [int(round(v*(frame-entry.last))) for v in entry.velocity]
                shifted = np.zeros_like(mask)
                h, w = mask.shape
                y0, y1 = max(0, -dy), min(h, h-dy)
                x0, x1 = max(0, -dx), min(w, w-dx)
                if y0 < y1 and x0 < x1:
                    shifted[y0+dy:y1+dy, x0+dx:x1+dx] = entry.mask[y0:y1, x0:x1]
                iou = (mask & shifted).sum()/max(1, (mask | shifted).sum())
                similarity = float(prototype @ entry.prototype)
                if iou >= .1 and similarity >= .4:
                    cost[a, b] = 1-(.7*iou+.3*similarity)
        matches = {}
        if cost.size:
            rows, columns = linear_sum_assignment(cost)
            for a, b in zip(rows, columns):
                if cost[a, b] < .8:
                    matches[int(a)] = old_ids[int(b)]
        output = np.zeros_like(labels)
        for a, (mask, prototype) in enumerate(regions):
            identity = matches.get(a)
            if identity is not None:
                entry = self.entries[identity]
                # Near-equal alternatives do not become a confirmed feature update.
                alternatives = np.sort(cost[a])
                ambiguous = len(alternatives) > 1 and alternatives[1]-alternatives[0] < .1
                if not ambiguous:
                    old_center = np.argwhere(entry.mask).mean(axis=0)
                    center = np.argwhere(mask).mean(axis=0)
                    dt = max(1, frame-entry.last)
                    velocity = tuple((center-old_center)/dt)
                    pooled = .9*entry.prototype+.1*prototype
                    pooled /= max(1e-8, np.linalg.norm(pooled))
                    self.entries[identity] = MemoryEntry(pooled, mask, frame, velocity)
            elif len(self.entries) < self.capacity:
                identity = self.next_id; self.next_id += 1
                self.entries[identity] = MemoryEntry(prototype, mask, frame)
            if identity is not None:
                output[mask] = identity
        return output
