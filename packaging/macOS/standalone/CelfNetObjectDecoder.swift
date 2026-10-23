import Foundation

/// Port of the o2 training decoder: hysteresis, centre votes, and orphan groups.
/// Pure Swift so the output and temporal input contracts can be checked separately.
enum CelfNetObjectDecoder {
  static let width = 240
  static let height = 136
  static let inputWidth = 960
  static let inputHeight = 544
  static let count = width * height

  struct Object {
    let pixels: [Int]
    let score: Float
  }

  static func sigmoid(_ x: Float) -> Float { 1 / (1 + exp(-x)) }

  static func neighbors(_ p: Int, diagonal: Bool = true) -> [Int] {
    let y = p / width, x = p % width
    var result: [Int] = []
    for dy in -1...1 {
      for dx in -1...1 where (dx != 0 || dy != 0)
        && (diagonal || abs(dx) + abs(dy) == 1)
      {
        let yy = y + dy, xx = x + dx
        if yy >= 0, yy < height, xx >= 0, xx < width {
          result.append(yy * width + xx)
        }
      }
    }
    return result
  }

  static func decode(_ output: [Float], confidence: Float = 0.5) -> [Object] {
    precondition(output.count == 11 * count)
    let prob = output[0..<count].map(sigmoid)
    let heat = output[(8 * count)..<(9 * count)].map(sigmoid)
    var component = [Int](repeating: 0, count: count)
    var regions: [[Int]] = [[]]
    // 8-connected hysteresis; weak regions survive only with a strong seed.
    for p in 0..<count where component[p] == 0 && prob[p] > min(confidence, 0.15) {
      let id = regions.count
      var queue = [p], cursor = 0, seeded = false
      component[p] = id
      while cursor < queue.count {
        let q = queue[cursor]; cursor += 1
        seeded = seeded || prob[q] > confidence
        for n in neighbors(q) where component[n] == 0 && prob[n] > min(confidence, 0.15) {
          component[n] = id; queue.append(n)
        }
      }
      if seeded { regions.append(queue) }
      else {
        // Mark visited weak background separately until all regions are found.
        for q in queue { component[q] = -1 }
      }
    }
    for p in 0..<count where component[p] < 0 { component[p] = 0 }
    var peaks: [Int] = []
    for p in 0..<count where heat[p] > 0.2 {
      let x = p % width, y = p / width
      var maximum = true
      for yy in max(0, y - 3)...min(height - 1, y + 3) {
        for xx in max(0, x - 3)...min(width - 1, x + 3) {
          if heat[yy * width + xx] > heat[p] { maximum = false; break }
        }
        if !maximum { break }
      }
      if maximum { peaks.append(p) }
    }
    let foreground = regions.dropFirst().flatMap { $0 }
    func distance(_ p: Int, _ peak: Int) -> Float {
      let vx = Float(p % width) + output[9 * count + p] * 16
      let vy = Float(p / width) + output[10 * count + p] * 16
      let dx = vx - Float(peak % width), dy = vy - Float(peak / width)
      return dx * dx + dy * dy
    }
    var home: [Int: [Int]] = [:]
    for (k, peak) in peaks.enumerated() {
      var c = component[peak]
      if c == 0 {
        var votes: [Int: Int] = [:]
        for p in foreground where distance(p, peak) < 16 {
          votes[component[p], default: 0] += 1
        }
        c = votes.keys.sorted().max { votes[$0]! < votes[$1]! } ?? 0
      }
      if c > 0 { home[c, default: []].append(k) }
    }
    func nearest(_ p: Int, _ candidates: [Int]) -> (Int, Float) {
      var best = candidates[0], d = distance(p, peaks[best])
      for k in candidates.dropFirst() {
        let dd = distance(p, peaks[k])
        if dd < d { best = k; d = dd }
      }
      return (best, d)
    }
    var owner = [Int](repeating: -1, count: count)
    var nextVirtual = peaks.count
    for c in 1..<regions.count {
      let pixels = regions[c]
      if pixels.count < 6 { continue }
      if let candidates = home[c], !candidates.isEmpty {
        var far: [Int] = []
        for p in pixels {
          let (k, d) = nearest(p, candidates); owner[p] = k
          if d > 64 { far.append(p) }
        }
        if far.count >= 40 {
          var sx: Double = 0, sy: Double = 0, sx2: Double = 0, sy2: Double = 0
          for p in far {
            let x = Double(Float(p % width) + output[9 * count + p] * 16)
            let y = Double(Float(p / width) + output[10 * count + p] * 16)
            sx += x; sy += y; sx2 += x * x; sy2 += y * y
          }
          let n = Double(far.count)
          if (sx2 + sy2) / n - (sx * sx + sy * sy) / (n * n) < 64 {
            for p in far { owner[p] = nextVirtual }; nextVirtual += 1
          }
        }
      } else {
        var votes: [Int: Int] = [:], close = 0
        if !peaks.isEmpty {
          for p in pixels {
            let (k, d) = nearest(p, Array(peaks.indices))
            if d < 16 { close += 1; votes[k, default: 0] += 1 }
          }
        }
        let k: Int
        if Float(close) / Float(pixels.count) >= 0.6 {
          k = votes.keys.sorted().max { votes[$0]! < votes[$1]! }!
        } else { k = nextVirtual; nextVirtual += 1 }
        for p in pixels { owner[p] = k }
      }
    }
    var groups: [Int: [Int]] = [:]
    for p in 0..<count where owner[p] >= 0 { groups[owner[p], default: []].append(p) }
    // Match the reference's unique ring pixels and smallest-owner tie break.
    let order = groups.keys.sorted { a, b in
      groups[a]!.count == groups[b]!.count ? a < b : groups[a]!.count < groups[b]!.count
    }
    for k in order {
      guard let pixels = groups[k], !pixels.isEmpty,
        pixels.count < (k >= peaks.count ? 40 : 20) else { continue }
      var ring = Set<Int>()
      for p in pixels {
        for q in neighbors(p) where owner[q] >= 0 && owner[q] != k { ring.insert(q) }
      }
      var votes: [Int: Int] = [:]
      for q in ring { votes[owner[q], default: 0] += 1 }
      if let target = votes.keys.sorted().max(by: { votes[$0]! < votes[$1]! }) {
        for p in pixels { owner[p] = target }
        groups[target, default: []].append(contentsOf: pixels); groups[k] = []
      }
    }
    return groups.keys.sorted().compactMap { k in
      let pixels = groups[k]!
      guard pixels.count >= 6 else { return nil }
      let score = pixels.reduce(Float(0)) { $0 + prob[$1] } / Float(pixels.count)
      return score >= 0.7 ? Object(pixels: pixels.sorted(), score: score) : nil
    }
  }

  /// Label maps use the same nearest 4x expansion and 2-row crop as training.
  static func inputLabels(_ objects: [Object]) -> [Int] {
    var small = [Int](repeating: 0, count: count)
    for (k, object) in objects.enumerated() {
      for p in object.pixels { small[p] = k + 1 }
    }
    var labels = [Int](repeating: 0, count: inputWidth * inputHeight)
    for y in 2..<542 {
      for x in 0..<inputWidth { labels[y * inputWidth + x] = small[(y / 4) * width + x / 4] }
    }
    return labels
  }

  /// Channel 6 stays zero; channel 7 is a twice-dilated boundary, 8 is centres.
  static func prior(_ labels: [Int]) -> [Float] {
    let w = inputWidth, h = inputHeight, n = w * h
    precondition(labels.count == n)
    var result = [Float](repeating: 0, count: 3 * n)
    var edge = [Bool](repeating: false, count: n)
    var sums: [Int: (x: Double, y: Double, n: Int)] = [:]
    for y in 0..<h {
      for x in 0..<w {
        let p = y * w + x, id = labels[p]
        if id == 0 { continue }
        let s = sums[id] ?? (0, 0, 0)
        sums[id] = (s.x + Double(x), s.y + Double(y), s.n + 1)
        for (dy, dx) in [(0, 1), (1, 0), (1, 1), (1, -1)] {
          let q = ((y - dy + h) % h) * w + ((x - dx + w) % w)
          if labels[q] > 0 && labels[q] != id { edge[p] = true }
        }
      }
    }
    for _ in 0..<2 {
      var expanded = edge
      for p in 0..<n where edge[p] {
        let x = p % w, y = p / w
        if x > 0 { expanded[p - 1] = true }; if x + 1 < w { expanded[p + 1] = true }
        if y > 0 { expanded[p - w] = true }; if y + 1 < h { expanded[p + w] = true }
      }
      edge = expanded
    }
    for p in 0..<n where edge[p] { result[n + p] = 1 }
    for s in sums.values {
      let cx = s.x / Double(s.n), cy = s.y / Double(s.n)
      for y in max(0, Int(cy - 24))..<min(h, Int(cy + 24) + 1) {
        for x in max(0, Int(cx - 24))..<min(w, Int(cx + 24) + 1) {
          let dx = Double(x) - cx, dy = Double(y) - cy
          let g = Float(exp(-(dx * dx + dy * dy) / 128))
          result[2 * n + y * w + x] = max(result[2 * n + y * w + x], g)
        }
      }
    }
    return result
  }
}
