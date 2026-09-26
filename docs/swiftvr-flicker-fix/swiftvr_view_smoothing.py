"""SwiftVR outline-flicker fix: smoothed crop view + evaluation tools.

Reference implementation (numpy + OpenCV only, platform independent) of the
fix described in README_ja.md. Frames are float32 RGB in [0, 1], HWC.

Geometry convention (lada-style clip restoration; adapt if yours differs):
  * A scene/clip has per-frame crop boxes (left, top, width, height) in the
    source frame.
  * Every crop is resized with ONE per-clip scale per axis,
        k_x = N / max_i(width_i),  k_y = N / max_i(height_i),  N = 256,
    to (w'_i, h'_i) = (floor(width_i*k_x), floor(height_i*k_y)) and centred in
    the N x N model grid with pad_i = ceil((N - w'_i) / 2) (same for y).
  * Grid pixels outside the resized crop are filled by REFLECTING the crop.
  * Inside the crop, grid centre g samples frame coordinate
        x = left_i + (g - pad_i + 0.5) * width_i / w'_i - 0.5,
    i.e. the linear map  g = s_i * x + o_i  with
        s_i = w'_i / width_i,   o_i = pad_i - 0.5 - (left_i - 0.5) * s_i.
"""
from dataclasses import dataclass
import numpy as np
import cv2

N = 256


@dataclass
class Crop:
    left: int
    top: int
    width: int
    height: int
    resized_w: int
    resized_h: int
    pad_left: int
    pad_top: int


def clip_crops(boxes, n=N):
    """boxes: list of (left, top, width, height) for one scene -> [Crop]."""
    kx = n / max(b[2] for b in boxes)
    ky = n / max(b[3] for b in boxes)
    out = []
    for l, t, w, h in boxes:
        rw, rh = max(1, int(w * kx)), max(1, int(h * ky))
        out.append(Crop(l, t, w, h, rw, rh, int(np.ceil((n - rw) / 2)), int(np.ceil((n - rh) / 2))))
    return out


def placement(c: Crop):
    """(s_x, o_x, s_y, o_y) of the map grid = s * frame + o."""
    sx, sy = c.resized_w / c.width, c.resized_h / c.height
    return (sx, c.pad_left - 0.5 - (c.left - 0.5) * sx,
            sy, c.pad_top - 0.5 - (c.top - 0.5) * sy)


def smooth_placements(placements, window=15):
    """Centred moving average over `window` frames, truncated at clip ends."""
    p = np.asarray(placements, np.float64)
    half = window // 2
    return [tuple(p[max(0, i - half):min(len(p), i + half + 1)].mean(0)) for i in range(len(p))]


def _remap(img, map_x, map_y):
    return cv2.remap(img, map_x.astype(np.float32), map_y.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def reframe(restored_grid, source_frame, crop: Crop, own, smooth, ramp=4.0):
    """SwiftVR input for one frame.

    restored_grid: N x N x 3 first-stage (e.g. BasicVSR++) output in this
                   frame's own grid (including its reflected padding).
    source_frame:  full source frame (H x W x 3) for the context outside the crop.
    own, smooth:   placement() of this frame and its smoothed placement.
    """
    sx, ox, sy, oy = own
    tsx, tox, tsy, toy = smooth
    g = np.arange(N, dtype=np.float64)
    fx, fy = (g - tox) / tsx, (g - toy) / tsy          # view grid -> frame
    gx, gy = fx * sx + ox, fy * sy + oy                # frame -> own grid
    mx, my = np.meshgrid(gx, gy)
    view = _remap(restored_grid, mx, my)
    # Restoration weight: 1 inside the crop, 0 outside, ramp over `ramp` px.
    def inside(pos, pad, count):
        d = np.minimum(pos - pad, pad + count - 1 - pos)
        return np.clip((d + 0.5) / ramp, 0, 1)
    w = np.outer(inside(gy, crop.pad_top, crop.resized_h), inside(gx, crop.pad_left, crop.resized_w))[..., None]
    fmx, fmy = np.meshgrid(fx, fy)
    context = _remap(source_frame, fmx, fmy)
    return (w * view + (1 - w) * context).astype(np.float32)


def map_back(swiftvr_out, own, smooth):
    """Map one SwiftVR output frame (M x M, M = N * scale) from the smoothed
    view back to this frame's own grid, before stabilization/compositing."""
    m = swiftvr_out.shape[0]
    k = m / N
    sx, ox, sy, oy = own
    tsx, tox, tsy, toy = smooth
    u = np.arange(m, dtype=np.float64)
    g = (u + 0.5) / k - 0.5
    vx = ((g - ox) / sx * tsx + tox + 0.5) * k - 0.5
    vy = ((g - oy) / sy * tsy + toy + 0.5) * k - 0.5
    mx, my = np.meshgrid(vx, vy)
    return _remap(swiftvr_out, mx, my)


# ---------------------------------------------------------------- probes ---

def probe_sequences(crop256, frames=48, seed=0, source=None, center=None, half=None):
    """Four 48-frame inputs from one restored 256px crop (float RGB).

    static: identical frames; noise: +N(0, 0.5/255); scale3: re-cropped from
    `source` with size jitter U(0.97, 1.03) around `center` (needs source,
    center=(cx, cy), half=crop half size); shift1: +-1 px sub-pixel shifts.
    """
    rng = np.random.default_rng(seed)
    seqs = {"static": [crop256] * frames,
            "noise": [np.clip(crop256 + rng.normal(0, 0.5 / 255, crop256.shape), 0, 1).astype(np.float32)
                      for _ in range(frames)]}
    shifts = []
    for _ in range(frames):
        dx, dy = rng.uniform(-1, 1, 2)
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        shifts.append(cv2.warpAffine(crop256, m, (N, N), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT))
    seqs["shift1"] = shifts
    if source is not None:
        cx, cy = center
        scaled = []
        for s in rng.uniform(0.97, 1.03, frames):
            h = int(half * s)
            scaled.append(cv2.resize(source[cy - h:cy + h, cx - h:cx + h], (N, N), interpolation=cv2.INTER_AREA))
        seqs["scale3"] = scaled
    return seqs


def added_detail_change(inputs, outputs):
    """Mean |delta_t (output - upscaled input)| in 8-bit levels: how much the
    detail SwiftVR adds changes from frame to frame."""
    m = outputs[0].shape[0]
    res = [o - cv2.resize(i, (m, m), interpolation=cv2.INTER_LINEAR) for i, o in zip(inputs, outputs)]
    return 255 * np.mean([np.abs(res[t] - res[t - 1]).mean() for t in range(1, len(res))])


# ---------------------------------------------------------------- metrics ---

def dense_mask(enhanced_luma, base_luma, box=15, threshold=3.0):
    """Where the enhancer actually changed the frame. Both inputs are 8-bit
    luma of separately ENCODED videos; a per-pixel threshold would also pick
    up scattered encoding differences all over the frame."""
    d = np.abs(enhanced_luma.astype(np.float32) - base_luma.astype(np.float32))
    return cv2.blur(d, (box, box)) > threshold


def flicker_increase(videos, base, mask_from, min_pixels=5000):
    """Temporal 2nd difference |X[t+1] - 2X[t] + X[t-1]| averaged in the dense
    mask, summed over frames, relative to the base (no enhancer) video.
    videos: {name: T x H x W uint8 luma}. Returns {name: percent}."""
    num = {k: 0.0 for k in videos}
    den = 0.0
    for t in range(1, len(base) - 1):
        m = dense_mask(mask_from[t], base[t])
        if m.sum() < min_pixels:
            continue
        sd = lambda x: np.abs(x[t + 1].astype(np.float32) - 2 * x[t] + x[t - 1])[m].mean()
        den += sd(base)
        for k, v in videos.items():
            num[k] += sd(v)
    return {k: (num[k] / den - 1) * 100 for k in videos}


def texture_ratio(videos, base, mask_from, step=3, min_pixels=5000):
    """1-3 px band energy |G_sigma1(X) - G_sigma3(X)| in the dense mask,
    relative to the base video (percent)."""
    acc = {k: 0.0 for k in videos}
    accb = 0.0
    band = lambda x, m: np.abs(cv2.GaussianBlur(x, (0, 0), 1) - cv2.GaussianBlur(x, (0, 0), 3))[m].mean()
    for t in range(0, len(base), step):
        m = dense_mask(mask_from[t], base[t])
        if m.sum() < min_pixels:
            continue
        accb += band(base[t].astype(np.float32), m)
        for k, v in videos.items():
            acc[k] += band(v[t].astype(np.float32), m)
    return {k: acc[k] / accb * 100 for k in videos}


if __name__ == "__main__":
    # Self-test: with an identity "enhancer" (output = 2x upscale of its
    # input), reframe -> map_back must return this frame's own grid inside the crop.
    rng = np.random.default_rng(1)
    frame = cv2.GaussianBlur(rng.random((1080, 1920, 3)).astype(np.float32), (0, 0), 3)
    boxes = [(800 + int(20 * np.sin(i / 3)), 400 + int(10 * np.cos(i / 4)),
              420 + int(40 * np.sin(i / 2)), 400 + int(30 * np.cos(i / 2))) for i in range(30)]
    crops = clip_crops(boxes)
    own = [placement(c) for c in crops]
    smooth = smooth_placements(own, 15)
    errs = []
    for i, c in enumerate(crops):
        g = np.arange(N, dtype=np.float64)
        mx, my = np.meshgrid((g - own[i][1]) / own[i][0], (g - own[i][3]) / own[i][2])
        grid = _remap(frame, mx, my)                      # this frame's own grid (no padding)
        view = reframe(grid, frame, c, own[i], smooth[i])
        out = cv2.resize(view, (2 * N, 2 * N), interpolation=cv2.INTER_LINEAR)
        back = cv2.resize(map_back(out, own[i], smooth[i]), (N, N), interpolation=cv2.INTER_AREA)
        sl = (slice(c.pad_top + 8, c.pad_top + c.resized_h - 8), slice(c.pad_left + 8, c.pad_left + c.resized_w - 8))
        errs.append(np.abs(back[sl] - grid[sl]).mean() * 255)
    print("round-trip error inside crops: mean %.3f, max %.3f levels" % (np.mean(errs), np.max(errs)))
