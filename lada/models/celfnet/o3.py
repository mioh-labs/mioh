"""Experimental o3: signed affinities, recurrent features and object memories.

Independent random initialization; no o2/CELF weights or labels are loaded.
This is not an app-compatible replacement for the 11-channel o2 model.
"""

import torch
from torch import nn
from torch.nn import functional as F

# Source -> destination in the stride-4 output. No wrapping at image borders.
OFFSETS = ((0, 1), (1, 0), (1, 1), (1, -1), (0, 4), (4, 0),
           (4, 4), (4, -4), (0, 8), (8, 0), (8, 8), (8, -8),
           (4, 8), (4, -8), (8, 4), (8, -4))


def block(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
        nn.GroupNorm(4, cout), nn.SiLU())


class O3AffinityNet(nn.Module):
    def __init__(self, channels=32, embedding_dim=16):
        super().__init__()
        self.channels, self.embedding_dim = channels, embedding_dim
        self.encoder = nn.Sequential(block(3, channels//2, 2), block(channels//2, channels, 2))
        self.context = nn.Sequential(block(channels, channels, 2), block(channels, channels, 2),
                                     block(channels, channels))
        self.query = nn.Conv2d(channels, embedding_dim, 1)
        self.memory_projection = nn.Conv2d(embedding_dim, channels, 1, bias=False)
        self.gates = nn.Conv2d(2*channels, 2*channels, 3, padding=1)
        self.proposal = nn.Conv2d(2*channels, channels, 3, padding=1)
        self.head = block(channels, channels)
        self.support = nn.Conv2d(channels, 1, 1)
        self.affinity = nn.Conv2d(channels, len(OFFSETS), 1)
        self.embedding = nn.Conv2d(channels, embedding_dim, 1)

    def forward(self, image, state=None, memory=None):
        features = self.encoder(image)
        features = features + F.interpolate(self.context(features), size=features.shape[-2:], mode="bilinear", align_corners=False)
        if state is None:
            state = torch.zeros_like(features)
        if state.shape != features.shape:
            raise ValueError("state geometry differs; reset at a size change or a cut")
        query = F.normalize(self.query(features), dim=1)
        if memory is not None and memory.shape[1] > 0:
            memory = F.normalize(memory, dim=-1)
            similarity = torch.einsum("bdhw,bkd->bkhw", query, memory)
            attention = (similarity*8).softmax(dim=1)
            recalled = torch.einsum("bkhw,bkd->bdhw", attention, memory)
            # An unrelated memory must not dominate the current image.
            confidence = ((similarity.max(dim=1, keepdim=True).values-.3)/.7).clamp(0, 1)
            features = features + self.memory_projection(recalled*confidence)
        reset, update = self.gates(torch.cat((features, state), dim=1)).sigmoid().chunk(2, dim=1)
        proposal = self.proposal(torch.cat((features, reset*state), dim=1)).tanh()
        state = (1-update)*state + update*proposal
        hidden = self.head(features+state)
        return dict(support=self.support(hidden), affinity=self.affinity(hidden),
                    embedding=F.normalize(self.embedding(hidden), dim=1), query=query, state=state)


def edge_targets(labels):
    """Same/different foreground ownership from exact procedural labels."""
    height, width = labels.shape[-2:]
    targets, valid = [], []
    for dy, dx in OFFSETS:
        shifted = torch.zeros_like(labels)
        ok = torch.zeros_like(labels, dtype=torch.bool)
        y0, y1 = max(0, -dy), min(height, height-dy)
        x0, x1 = max(0, -dx), min(width, width-dx)
        shifted[..., y0:y1, x0:x1] = labels[..., y0+dy:y1+dy, x0+dx:x1+dx]
        ok[..., y0:y1, x0:x1] = True
        valid.append(ok & (labels > 0) & (shifted > 0))
        targets.append((labels == shifted).float())
    return torch.stack(targets, dim=-3), torch.stack(valid, dim=-3)


def supervision(output, labels, previous_prototypes=None):
    """Ground truth is used in the loss, never as next-frame model input."""
    target = (labels > 0).float().unsqueeze(1)
    # Background occupies most pixels. Balance foreground without making empty
    # clips positive examples or handing their truth to recurrent state.
    positives = target.sum()
    positive_weight = ((target.numel()-positives)/positives.clamp_min(1)).clamp(1, 12)
    loss_support = F.binary_cross_entropy_with_logits(output["support"], target, pos_weight=positive_weight)
    probability = output["support"].sigmoid()
    loss_support = loss_support + 1-(2*(probability*target).sum()+1)/(probability.sum()+target.sum()+1)
    relation, valid = edge_targets(labels)
    edge_loss = F.binary_cross_entropy_with_logits(output["affinity"], relation, reduction="none")
    positive, negative = valid & (relation > .5), valid & (relation < .5)
    loss_relation = (edge_loss*positive).sum()/positive.sum().clamp_min(1)
    loss_relation = loss_relation + (edge_loss*negative).sum()/negative.sum().clamp_min(1)
    # Object IDs are clip-local, not a global semantic class or a fixed count.
    embedding = output["embedding"][0]
    prototypes = {}
    cohesion = embedding.sum()*0
    temporal = cohesion
    identities = torch.unique(labels[0]).tolist()
    for identity in identities:
        if identity == 0:
            continue
        pixels = embedding[:, labels[0] == identity]
        prototype = F.normalize(pixels.mean(dim=1), dim=0)
        prototypes[int(identity)] = prototype
        cohesion = cohesion + (1-(pixels*prototype[:, None]).sum(dim=0)).mean()
        if previous_prototypes is not None and int(identity) in previous_prototypes:
            temporal = temporal + 1-(prototype*previous_prototypes[int(identity)]).sum()
    separation = embedding.sum()*0
    values = list(prototypes.values())
    for i, a in enumerate(values):
        for b in values[i+1:]:
            separation = separation + F.relu((a*b).sum()-.2).square()
    count = max(1, len(values))
    loss_embedding = (cohesion+temporal+separation)/count
    # Memory keys come from the output embedding; pre-read queries must live
    # in the SAME feature space. This is supervision, not teacher-fed memory.
    query_alignment = embedding.sum()*0
    for identity, prototype in prototypes.items():
        query_pixels = output["query"][0, :, labels[0] == identity]
        own = (query_pixels*prototype.detach()[:, None]).sum(dim=0)
        query_alignment = query_alignment + (1-own).mean()
        for other_identity, other in prototypes.items():
            if other_identity != identity:
                wrong = (query_pixels*other.detach()[:, None]).sum(dim=0)
                query_alignment = query_alignment + F.relu(wrong-own+.2).mean()/max(1,count-1)
    query_alignment = query_alignment/count
    total = loss_support + 2*loss_relation + .3*loss_embedding + .3*query_alignment
    return total, prototypes, dict(support=loss_support.detach(), relation=loss_relation.detach(),
        embedding=loss_embedding.detach(), query_alignment=query_alignment.detach())
