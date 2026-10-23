"""Object CELF-Net outputs -> mosaic objects (one instance per object).

Mosaic pixels: hysteresis on the mosaic probability (as decode.py).
Objects: peaks of the centre map; every mosaic pixel votes for a centre
(its position + predicted offset) and joins the nearest peak of its own
connected region. Rules that keep separate mosaics separate and one
mosaic whole:
  - a region without a peak of its own is one object, unless most of its
    votes land on another region's peak (one object cut by a finger);
  - pixels whose votes land far from every peak of their region (a
    missed centre) form an object of their own when there are enough of
    them;
  - tiny groups inside a region go to the neighbour they touch most;
"""
import numpy as np
from scipy import ndimage

from .model import OFFSET_SCALE
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -80, 80)))


def decode_objects(out, conf=0.5, extent=0.15, min_px=6, min_score=0.7, centre_thr=0.2, nms=7,
                   r_join=4.0, r_miss=8.0, min_group=40):
    prob = sigmoid(out[0])
    heat = sigmoid(out[8])
    ox, oy = out[9] * OFFSET_SCALE, out[10] * OFFSET_SCALE
    seeds = prob > conf
    wide, nw = ndimage.label(prob > min(conf, extent), structure=np.ones((3, 3)))
    keep = np.zeros(nw + 1, bool)
    keep[np.unique(wide[seeds])] = True
    keep[0] = False
    fg = keep[wide]
    comps, n = ndimage.label(fg, structure=np.ones((3, 3)))
    H, W = fg.shape
    hmax = ndimage.maximum_filter(heat, size=nms)
    py, px = np.nonzero((heat == hmax) & (heat > centre_thr))
    peaks = [(float(y), float(x)) for y, x in zip(py, px)]

    ys, xs = np.nonzero(fg)
    vy, vx = ys + oy[ys, xs], xs + ox[ys, xs]
    pc = comps[ys, xs]

    def nearest(cands, sel):
        """For pixels sel (indices into ys), nearest of the candidate peaks
        by vote: (index into cands, distance)."""
        P = np.array([peaks[c] for c in cands])
        d2 = (vy[sel, None] - P[None, :, 0]) ** 2 + (vx[sel, None] - P[None, :, 1]) ** 2
        j = d2.argmin(1)
        return np.array(cands)[j], np.sqrt(d2[np.arange(len(sel)), j])

    # Home region of every peak: the region it sits in, else the region
    # whose votes land closest to it (a curved object's centre can lie off it).
    home = {}
    for k, (y, x) in enumerate(peaks):
        c = comps[int(y), int(x)]
        if c == 0 and len(ys):
            d2 = (vy - y) ** 2 + (vx - x) ** 2
            close = d2 < r_join ** 2
            if close.any():
                c = int(np.bincount(pc[close]).argmax())
        if c:
            home.setdefault(c, []).append(k)

    owner = np.full(len(ys), -1, np.int64)     # peak index, or -(region) - 2 for peakless objects
    next_virtual = len(peaks)
    for c in range(1, n + 1):
        sel = np.nonzero(pc == c)[0]
        if len(sel) < min_px:
            continue
        cands = home.get(c, [])
        if cands:
            j, d = nearest(cands, sel)
            owner[sel] = j
            far = sel[d > r_miss]
            if len(far) >= min_group and np.sqrt(vy[far].var() + vx[far].var()) < r_miss:
                # A centre the map missed: far voters that agree on one place
                # are an object of their own (scattered ones sit on a border
                # between two objects and stay with their nearest centre).
                owner[far] = next_virtual; next_virtual += 1
            continue
        # No peak of its own: part of an object elsewhere, or an object itself.
        if peaks:
            j, d = nearest(list(range(len(peaks))), sel)
            close = d < r_join
            if close.mean() >= 0.6:
                owner[sel] = int(np.bincount(j[close]).argmax())
                continue
        owner[sel] = next_virtual; next_virtual += 1

    inst = np.zeros((H, W), np.int32)
    ids = {}
    for o in np.unique(owner):
        if o < 0:
            continue
        ids[int(o)] = len(ids) + 1
    for o, i in ids.items():
        m = owner == o
        inst[ys[m], xs[m]] = i
    # Tiny groups of a region go to the neighbour they touch most (a group
    # with a centre of its own only when it is very small).
    virtual = {i for o, i in ids.items() if o >= len(peaks)}
    sizes = np.bincount(inst.ravel(), minlength=len(ids) + 1)
    for k in sorted(range(1, len(ids) + 1), key=lambda k: sizes[k]):
        m = inst == k
        size = int(m.sum())
        if size == 0 or size >= (min_group if k in virtual else 20):
            continue
        ring = inst[ndimage.binary_dilation(m, structure=np.ones((3, 3))) & ~m]
        ring = ring[ring > 0]
        if len(ring):
            inst[m] = int(np.bincount(ring).argmax())
    result = []
    for k in range(1, len(ids) + 1):
        m = inst == k
        if m.sum() >= min_px and float(prob[m].mean()) >= min_score:
            result.append(dict(mask=m, score=float(prob[m].mean())))
    return result
