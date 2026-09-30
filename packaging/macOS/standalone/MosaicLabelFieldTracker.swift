import Foundation

// CELF (Claim-Evidence Label Field) mosaic tracking.
//
// Detector boxes carry no identity. Every mosaic pixel (on a 4 px cell grid)
// has exactly one owner, carried from frame to frame: the previous owners,
// shifted by each track's motion, compete to cover the union of the detector
// masks. Merged, straddling and duplicate boxes only say "mosaic is here", so
// they can neither create, end nor move a track. Where two neighbouring
// mosaics were pixelated with different grids (cell size/phase), the grid
// keeps their border; where the detector keeps seeing two separate mosaics
// inside one track, the track splits. Each track's crop then comes from its
// own cells only, so a neighbour can never enlarge it, and the paste weights
// come from one owner map, so no pixel is pasted twice.
//
// Ported from the Python prototype (labelprop.py v6) that kept 4/4/4 tracks
// with 0/4/0 count changes (the 4 were real occlusions) on three test clips.

struct LabelFieldBox: Equatable, Sendable {
  var left: Int
  var top: Int
  var right: Int
  var bottom: Int

  var width: Int { right - left + 1 }
  var height: Int { bottom - top + 1 }
  var area: Int { max(0, width) * max(0, height) }

  func intersectionArea(_ other: LabelFieldBox) -> Int {
    let width = min(right, other.right) - max(left, other.left) + 1
    let height = min(bottom, other.bottom) - max(top, other.top) + 1
    return width > 0 && height > 0 ? width * height : 0
  }
}

/// A detector mask on the tracker's cell grid.
struct LabelFieldObservation: Sendable {
  let box: LabelFieldBox
  /// Cell indices (y * gridWidth + x) the detector marks as mosaic.
  let cells: [Int]
}

/// The owner of every mosaic cell for one frame and how much of the
/// owner's restoration covers it: 1 on the owner's cells and a margin around
/// them, then a smoothstep fade (the same margin and feather as the
/// per-detection mask-only paste), so the restoration covers the mosaic edge
/// the detector mask misses. Only the rectangle holding owned cells is
/// stored, as 16-bit ids and 8-bit coverage, since a batch keeps hundreds of
/// these alive.
final class LabelFieldFrame: @unchecked Sendable {
  let gridWidth: Int
  let gridHeight: Int
  let cellSize: Int
  /// The stored cell rectangle.
  let originX: Int
  let originY: Int
  let width: Int
  let height: Int
  private let owners: [UInt16]
  private let coverage: [UInt8]
  /// Pixel bounds of each track's cells.
  let boxes: [Int: LabelFieldBox]

  /// - Parameters:
  ///   - paste: owner per cell of the whole grid (0 = none).
  ///   - coverage: the owner's weight per cell of the whole grid (0...1).
  convenience init(
    gridWidth: Int, gridHeight: Int, cellSize: Int, paste: [Int32], coverage: [Float]
  ) {
    var bounds: (top: Int, bottom: Int, left: Int, right: Int)?
    var perTrack: [Int32: (top: Int, bottom: Int, left: Int, right: Int)] = [:]
    for cell in paste.indices where paste[cell] > 0 {
      let y = cell / gridWidth
      let x = cell % gridWidth
      bounds = bounds.map {
        (min($0.top, y), max($0.bottom, y), min($0.left, x), max($0.right, x))
      } ?? (y, y, x, x)
      perTrack[paste[cell]] = perTrack[paste[cell]].map {
        (min($0.top, y), max($0.bottom, y), min($0.left, x), max($0.right, x))
      } ?? (y, y, x, x)
    }
    let rect = bounds ?? (0, -1, 0, -1)
    let width = max(0, rect.right - rect.left + 1)
    let height = max(0, rect.bottom - rect.top + 1)
    var owners = [UInt16](repeating: 0, count: width * height)
    var weights = [UInt8](repeating: 0, count: width * height)
    for y in 0..<height {
      for x in 0..<width {
        let cell = (rect.top + y) * gridWidth + rect.left + x
        owners[y * width + x] = UInt16(clamping: paste[cell])
        weights[y * width + x] = paste[cell] > 0
          ? UInt8((min(1, max(0, coverage[cell])) * 255).rounded()) : 0
      }
    }
    var boxes: [Int: LabelFieldBox] = [:]
    for (id, box) in perTrack {
      boxes[Int(id)] = LabelFieldBox(
        left: box.left * cellSize, top: box.top * cellSize,
        right: (box.right + 1) * cellSize - 1, bottom: (box.bottom + 1) * cellSize - 1)
    }
    self.init(
      gridWidth: gridWidth, gridHeight: gridHeight, cellSize: cellSize,
      originX: rect.left, originY: rect.top, width: width, height: height,
      owners: owners, coverage: weights, boxes: boxes)
  }

  private init(
    gridWidth: Int, gridHeight: Int, cellSize: Int,
    originX: Int, originY: Int, width: Int, height: Int,
    owners: [UInt16], coverage: [UInt8], boxes: [Int: LabelFieldBox]
  ) {
    self.gridWidth = gridWidth
    self.gridHeight = gridHeight
    self.cellSize = cellSize
    self.originX = originX
    self.originY = originY
    self.width = width
    self.height = height
    self.owners = owners
    self.coverage = coverage
    self.boxes = boxes
  }

  /// Owner of a grid cell (0 = none).
  @inline(__always)
  func owner(cellX: Int, cellY: Int) -> Int32 {
    let x = cellX - originX
    let y = cellY - originY
    guard x >= 0, x < width, y >= 0, y < height else { return 0 }
    return Int32(owners[y * width + x])
  }

  /// Owner and its weight (0...1) at a grid cell.
  @inline(__always)
  func ownerAndWeight(cellX: Int, cellY: Int) -> (Int32, Float) {
    let x = cellX - originX
    let y = cellY - originY
    guard x >= 0, x < width, y >= 0, y < height else { return (0, 0) }
    let index = y * width + x
    return (Int32(owners[index]), Float(coverage[index]) / 255)
  }

  @inline(__always)
  func pasteOwner(x: Int, y: Int) -> Int32 {
    owner(cellX: x / cellSize, cellY: y / cellSize)
  }

  /// Grid cells (y * gridWidth + x) a track owns, with their weights.
  func cells(of id: Int) -> [(cell: Int, weight: Float)] {
    var result: [(cell: Int, weight: Float)] = []
    for y in 0..<height {
      for x in 0..<width where owners[y * width + x] == UInt16(clamping: id) {
        result.append(((originY + y) * gridWidth + originX + x, Float(coverage[y * width + x]) / 255))
      }
    }
    return result
  }

  /// A copy in which each given track also owns the given cells wherever
  /// nobody owns them yet (a mosaic the detector missed for a few frames).
  func filling(_ additions: [Int: [(cell: Int, weight: Float)]]) -> LabelFieldFrame {
    var paste = [Int32](repeating: 0, count: gridWidth * gridHeight)
    var weights = [Float](repeating: 0, count: gridWidth * gridHeight)
    for y in 0..<height {
      for x in 0..<width {
        let cell = (originY + y) * gridWidth + originX + x
        paste[cell] = Int32(owners[y * width + x])
        weights[cell] = Float(coverage[y * width + x]) / 255
      }
    }
    for id in additions.keys.sorted() {
      for addition in additions[id]!
      where addition.cell >= 0 && addition.cell < paste.count && paste[addition.cell] == 0 {
        paste[addition.cell] = Int32(id)
        weights[addition.cell] = addition.weight
      }
    }
    return LabelFieldFrame(
      gridWidth: gridWidth, gridHeight: gridHeight, cellSize: cellSize,
      paste: paste, coverage: weights)
  }
}

final class MosaicLabelFieldTracker: @unchecked Sendable {
  static let cellSize = 4

  struct Lattice {
    var px: Double
    var phx: Double
    var py: Double
    var phy: Double
    var r: Double
  }

  private final class Track {
    let id: Int
    let born: Int
    var last: Int
    var lattice: Lattice?
    var cells: [Int] = []
    var vy = 0
    var vx = 0
    var miss = 0
    var carryAge = 0
    var mismatch = 0
    var splitPending = 0
    var detached = 0
    var lastBox: LabelFieldBox?
    var claimSplitHistory: [Bool] = []

    init(id: Int, frame: Int, lattice: Lattice?) {
      self.id = id
      born = frame
      last = frame
      self.lattice = lattice
    }
  }

  // Thresholds from the prototype (labelprop.py CFG).
  private let alpha: Float = 4
  private let beta: Float = 3
  /// CELF: entering a cell that only another track's detection claims.
  private let claimCost: Float = 2
  private let gateFraction = 0.3
  private let gateMinimum = 6
  private let carryScore: Float = 1
  private let carryMaximum = 15
  private let minimumCells = 30
  private let lostKeep = 90
  private let missMaximum = 10
  private let auditEvery = 5
  private let reidPixels = 64

  let width: Int
  let height: Int
  let gridWidth: Int
  let gridHeight: Int
  private let cellCount: Int
  private let windowSize = 64
  private let windowStride = 16
  private let windowRows: Int
  private let windowColumns: Int
  private let cellWindowRow: [Int]
  private let cellWindowColumn: [Int]

  /// Cells each track was pasted over lately (own or held) and for how many
  /// frames they have been held without an owner.
  private var shown: [Int: [(cell: Int, age: Int)]] = [:]
  /// The detector reacts a frame or two late when a mosaic slips behind
  /// something: the last sliver is still visible but no longer detected.
  /// Cells a track just lost keep its restoration this many frames.
  private let holdFrames = 2

  private var tracks: [Int: Track] = [:]
  private var lost: [Int: Track] = [:]
  private var nextID = 1
  private var frameIndex = -1
  private var previousThumbnail: [UInt8]?

  /// The per-detection paste's feather factor (fade = 5% of the crop's
  /// short side times this).
  private let feather: Double

  init(width: Int, height: Int, feather: Double = 1) {
    self.width = width
    self.height = height
    self.feather = max(0, feather)
    gridWidth = (width + Self.cellSize - 1) / Self.cellSize
    gridHeight = (height + Self.cellSize - 1) / Self.cellSize
    cellCount = gridWidth * gridHeight
    windowRows = max(1, (height - windowSize) / windowStride + 1)
    windowColumns = max(1, (width - windowSize) / windowStride + 1)
    let rows = windowRows
    let columns = windowColumns
    let stride = windowStride
    let size = windowSize
    cellWindowRow = (0..<gridHeight).map {
      min(rows - 1, max(0, Int((Double($0 * Self.cellSize + 2 - size / 2)
        / Double(stride)).rounded())))
    }
    cellWindowColumn = (0..<gridWidth).map {
      min(columns - 1, max(0, Int((Double($0 * Self.cellSize + 2 - size / 2)
        / Double(stride)).rounded())))
    }
  }

  // MARK: - Frame update

  /// - Parameters:
  ///   - observations: this frame's detector masks on the cell grid.
  ///   - luma: 8-bit luma of a pixel rectangle (row-major, width x height).
  ///   - thumbnail: a small fixed-size luma thumbnail for cut detection.
  func update(
    observations: [LabelFieldObservation],
    luma: (LabelFieldBox) -> [UInt8],
    thumbnail: [UInt8]?
  ) -> LabelFieldFrame {
    frameIndex += 1
    let i = frameIndex
    if let thumbnail, let previousThumbnail,
      thumbnail.count == previousThumbnail.count, !thumbnail.isEmpty
    {
      var total = 0
      for index in thumbnail.indices {
        total += abs(Int(thumbnail[index]) - Int(previousThumbnail[index]))
      }
      // A hard cut: nothing on screen continues, so no identity may either.
      if Double(total) / Double(thumbnail.count) > 40 {
        tracks.removeAll()
        lost.removeAll()
        shown.removeAll()
      }
    }
    previousThumbnail = thumbnail
    lost = lost.filter { i - $0.value.last <= lostKeep }

    var support = [Bool](repeating: false, count: cellCount)
    for observation in observations {
      for cell in observation.cells { support[cell] = true }
    }
    let detectorSupport = support

    // Predict each track: translation search around its previous velocity.
    let ids = tracks.keys.sorted()
    var predicted: [Int: [Bool]] = [:]
    var predictedCells: [Int: [Int]] = [:]
    var predictedVelocity: [Int: (Int, Int)] = [:]
    for id in ids {
      let track = tracks[id]!
      guard !track.cells.isEmpty else { continue }
      let step = max(1, track.cells.count / 3000)
      var bestScore = -1.0
      var best = (track.vy, track.vx)
      for dy in -3...3 {
        for dx in -3...3 {
          let vy = track.vy + dy
          let vx = track.vx + dx
          var hits = 0
          var index = 0
          while index < track.cells.count {
            let cell = track.cells[index]
            let y = min(gridHeight - 1, max(0, cell / gridWidth + vy))
            let x = min(gridWidth - 1, max(0, cell % gridWidth + vx))
            if support[y * gridWidth + x] { hits += 1 }
            index += step
          }
          let score = Double(hits) - 0.01 * Double(abs(dy) + abs(dx))
          if score > bestScore {
            bestScore = score
            best = (vy, vx)
          }
        }
      }
      if bestScore <= 0 { best = (0, 0) }
      var mask = [Bool](repeating: false, count: cellCount)
      var shifted: [Int] = []
      shifted.reserveCapacity(track.cells.count)
      for cell in track.cells {
        let y = min(gridHeight - 1, max(0, cell / gridWidth + best.0))
        let x = min(gridWidth - 1, max(0, cell % gridWidth + best.1))
        let target = y * gridWidth + x
        if !mask[target] {
          mask[target] = true
          shifted.append(target)
        }
      }
      predicted[id] = mask
      predictedCells[id] = shifted
      predictedVelocity[id] = best
    }

    // Luma edges inside the eroded (detections + predictions) support.
    var gradientSupport = support
    for id in predicted.keys {
      for cell in predictedCells[id]! { gradientSupport[cell] = true }
    }
    let edges = edgeMaps(support: gradientSupport, luma: luma)

    // Grid (lattice) scores of every track that knows its grid.
    var lattices: [Int: Lattice] = [:]
    for id in ids {
      if let lattice = tracks[id]!.lattice, lattice.r >= 0.55 {
        lattices[id] = lattice
      }
    }
    let windowScores = edges.flatMap { self.windowScores($0, lattices: lattices) }
    var cellScores: [Int: [Float]] = [:]
    var confident = [Bool](repeating: false, count: cellCount)
    if let edges, let windowScores {
      for cell in 0..<cellCount {
        let window = cellWindowRow[cell / gridWidth] * windowColumns
          + cellWindowColumn[cell % gridWidth]
        confident[cell] = windowScores.confident[window]
          && edges.coversCell(cell, gridWidth: gridWidth)
      }
      for (id, scores) in windowScores.scores {
        var map = [Float](repeating: -9, count: cellCount)
        for cell in 0..<cellCount where confident[cell] {
          map[cell] = scores[cellWindowRow[cell / gridWidth] * windowColumns
            + cellWindowColumn[cell % gridWidth]]
        }
        cellScores[id] = map
      }
    }

    // A briefly undetected mosaic that its own grid still explains stays.
    for id in ids {
      guard let track = tracks[id], let cells = predictedCells[id],
        let scores = cellScores[id], track.carryAge < carryMaximum
      else { continue }
      for cell in cells
      where !support[cell] && confident[cell] && scores[cell] >= carryScore {
        support[cell] = true
      }
    }

    // Near regions, best competing grid score, novelty and extensions.
    let activeIDs = ids.filter { predicted[$0] != nil }
    var nearBoxes: [Int: (top: Int, bottom: Int, left: Int, right: Int)] = [:]
    for id in activeIDs {
      nearBoxes[id] = cellBounds(predictedCells[id]!)
    }
    var bestScore = [Float](repeating: -9, count: cellCount)
    for id in activeIDs {
      guard let scores = cellScores[id], let bounds = nearBoxes[id] else { continue }
      for y in max(0, bounds.top - 38)...min(gridHeight - 1, bounds.bottom + 38) {
        for x in max(0, bounds.left - 38)...min(gridWidth - 1, bounds.right + 38) {
          let cell = y * gridWidth + x
          bestScore[cell] = max(bestScore[cell], scores[cell])
        }
      }
    }
    var predictedUnion = [Bool](repeating: false, count: cellCount)
    for id in activeIDs {
      for cell in predictedCells[id]! { predictedUnion[cell] = true }
    }
    let predictedUnionDilated = dilate(predictedUnion, radius: 1)
    var predictedDilated: [Int: [Bool]] = [:]
    for id in activeIDs {
      predictedDilated[id] = dilate(predicted[id]!, radius: 2)
    }
    var novel = [Bool](repeating: false, count: cellCount)
    var extend: [Int: [Bool]] = [:]
    for observation in observations {
      let count = observation.cells.count
      guard count >= minimumCells else { continue }
      let inPredicted = observation.cells.reduce(0) { $0 + (predictedUnion[$1] ? 1 : 0) }
      if Double(inPredicted) >= 0.5 * Double(count) { continue }
      let part = observation.cells.filter { !predictedUnionDilated[$0] }
      let touching = activeIDs.filter { id in
        observation.cells.contains { predictedDilated[id]![$0] }
      }
      if touching.isEmpty {
        for cell in part { novel[cell] = true }
        continue
      }
      let confidentPart = part.filter { confident[$0] }
      var explained: [(Int, Float)] = []
      if confidentPart.count >= 8 {
        for id in touching {
          if let scores = cellScores[id] {
            explained.append((id, median(confidentPart.map { scores[$0] })))
          }
        }
      }
      if let best = explained.map(\.1).max(), best < 0.3,
        confidentPart.count >= minimumCells,
        let edges, let own = fitRegion(part, edges: edges),
        touching.allSatisfy({ differentGrid(own, tracks[$0]?.lattice) })
      {
        // A different grid right next to the track is a new mosaic. Not
        // explaining the part is not enough: an occluded or blurred piece
        // of the same mosaic does not fit either, so its own grid must be
        // measured and differ.
        for cell in part { novel[cell] = true }
      } else {
        let owner = explained.max { lhs, rhs in
          lhs.1 != rhs.1 ? lhs.1 < rhs.1 : lhs.0 > rhs.0
        }?.0 ?? touching[0]
        var mask = extend[owner] ?? [Bool](repeating: false, count: cellCount)
        for cell in observation.cells { mask[cell] = true }
        extend[owner] = mask
      }
    }

    // CELF claims: a detection over exactly one track is that track's
    // evidence; one over several tracks (or around several detections) is
    // only support.
    let covers = coverObservations(observations)
    var claimOwner = [Int32](repeating: 0, count: cellCount)
    var claimsByTrack: [Int: [Int]] = [:]
    let claimReach = Dictionary(uniqueKeysWithValues: activeIDs.map {
      ($0, dilate(predicted[$0]!, radius: 3))
    })
    for (index, observation) in observations.enumerated()
    where !covers.contains(index) && !observation.cells.isEmpty {
      let count = Double(observation.cells.count)
      let sharing = activeIDs.filter { id in
        let shared = observation.cells.reduce(0) { $0 + (predicted[id]![$1] ? 1 : 0) }
        return Double(shared) >= 0.2 * count
      }
      guard sharing.count == 1, let id = sharing.first else { continue }
      claimsByTrack[id, default: []].append(index)
      let reach = claimReach[id]!
      for cell in observation.cells where reach[cell] {
        let slot = Int32(id)
        if claimOwner[cell] == 0 {
          claimOwner[cell] = slot
        } else if claimOwner[cell] != slot {
          claimOwner[cell] = -1
        }
      }
    }

    // Seeds, gates and grid mismatch per track, then competitive growth.
    var slots: [Int] = []
    var seeds = [Int32](repeating: 0, count: cellCount)
    var previous = [Int32](repeating: 0, count: cellCount)
    var conflict = [Bool](repeating: false, count: cellCount)
    var gates: [[Bool]] = []
    var mismatches: [[Float]?] = []
    for id in activeIDs {
      let slot = Int32(slots.count + 1)
      slots.append(id)
      let bounds = nearBoxes[id]!
      let height = bounds.bottom - bounds.top + 1
      let width = bounds.right - bounds.left + 1
      let gateY = max(gateMinimum, Int(gateFraction * Double(height)))
      let gateX = max(gateMinimum, Int(gateFraction * Double(width)))
      var gate = [Bool](repeating: false, count: cellCount)
      for y in max(0, bounds.top - gateY)...min(gridHeight - 1, bounds.bottom + gateY) {
        for x in max(0, bounds.left - gateX)...min(gridWidth - 1, bounds.right + gateX) {
          gate[y * gridWidth + x] = true
        }
      }
      if let extended = extend[id] {
        for cell in 0..<cellCount where extended[cell] { gate[cell] = true }
      }
      for cell in 0..<cellCount where novel[cell] { gate[cell] = false }
      gates.append(gate)
      var mismatch: [Float]?
      if let scores = cellScores[id] {
        var map = [Float](repeating: 0, count: cellCount)
        for cell in 0..<cellCount
        where confident[cell] && bestScore[cell] > 0.6 {
          map[cell] = max(0, bestScore[cell] - scores[cell])
        }
        mismatch = map
      }
      mismatches.append(mismatch)
      let eroded = erode(predicted[id]!, radius: 1)
      for cell in predictedCells[id]! {
        if previous[cell] == 0 { previous[cell] = slot }
        guard eroded[cell], support[cell] else { continue }
        if let scores = cellScores[id], confident[cell], bestScore[cell] > 0.6,
          bestScore[cell] - scores[cell] > 0.4
        {
          continue
        }
        // CELF: another track's detection alone claims this cell.
        if claimOwner[cell] > 0 && claimOwner[cell] != Int32(id) { continue }
        if seeds[cell] > 0 { conflict[cell] = true }
        seeds[cell] = slot
      }
    }
    for cell in 0..<cellCount where conflict[cell] { seeds[cell] = 0 }
    let claimSlots = claimOwner.map { owner -> Int32 in
      guard owner > 0, let slot = slots.firstIndex(of: Int(owner)) else {
        return 0
      }
      return Int32(slot + 1)
    }
    var labels = [Int32](repeating: 0, count: cellCount)
    if !slots.isEmpty {
      let grown = grow(
        region: support, seeds: seeds, mismatches: mismatches,
        previous: previous, gates: gates, claims: claimSlots)
      for cell in 0..<cellCount where grown[cell] > 0 {
        labels[cell] = Int32(slots[Int(grown[cell]) - 1])
      }
    }

    // Orphans: re-identify a lost track or start one (split by grid).
    let orphanMask = (0..<cellCount).map { support[$0] && labels[$0] == 0 }
    for component in components(orphanMask, eightConnected: false)
    where component.count >= minimumCells {
      resolveOrphan(
        component, frame: i, labels: &labels, edges: edges,
        cellScores: cellScores, confident: confident)
    }

    // Audit: a track whose own region shows two grids is two mosaics.
    if let edges {
      for id in tracks.keys.sorted() {
        guard let track = tracks[id], (i - track.born) % auditEvery == 0 else { continue }
        auditSplit(track, frame: i, labels: &labels, edges: edges, confident: confident)
      }
    }

    // CELF: two separate detections inside one track, again and again.
    claimSplit(
      frame: i, labels: &labels, observations: observations,
      claimsByTrack: claimsByTrack, predicted: predicted)

    reattachDetachedPieces(labels: &labels, cellScores: cellScores, confident: confident)

    // Small leftover specks go to the nearest label within 16 cells; a
    // piece farther from every track is still mosaic and becomes a track of
    // its own, so nothing the detector marks is left unrestored.
    if labels.contains(where: { $0 != 0 }) {
      let nearest = nearestLabels(labels, maximumDistance: 16)
      for cell in 0..<cellCount where support[cell] && labels[cell] == 0 {
        labels[cell] = nearest[cell]
      }
    }
    let isolated = (0..<cellCount).map { support[$0] && labels[$0] == 0 }
    for piece in components(isolated, eightConnected: true) {
      let box = pixelBox(cellBounds(piece))
      let track = reidentify(box, lattice: nil, frame: i) ?? newTrack(frame: i, lattice: nil)
      for cell in piece { labels[cell] = Int32(track.id) }
    }

    // Update the tracks from their cells.
    var cellsByLabel: [Int32: [Int]] = [:]
    for cell in 0..<cellCount where labels[cell] != 0 {
      cellsByLabel[labels[cell], default: []].append(cell)
    }
    for id in tracks.keys.sorted() {
      guard let track = tracks[id] else { continue }
      let cells = cellsByLabel[Int32(id)] ?? []
      if cells.count >= minimumCells {
        let bounds = cellBounds(cells)
        let box = pixelBox(bounds)
        let detected = cells.reduce(0) { $0 + (detectorSupport[$1] ? 1 : 0) }
        track.carryAge = Double(detected) < 0.3 * Double(cells.count)
          ? track.carryAge + 1 : 0
        if let velocity = predictedVelocity[id] {
          track.vy = velocity.0
          track.vx = velocity.1
        }
        track.cells = cells
        track.last = i
        track.miss = 0
        track.lastBox = box
        if (i - track.born) % 2 == 0, let edges,
          let fit = fitRegion(cells, edges: edges), fit.r >= 0.6
        {
          if let lattice = track.lattice, lattice.r >= 0.55,
            abs(fit.px - lattice.px) >= 1.5 || abs(fit.py - lattice.py) >= 1.5
          {
            track.mismatch += 1
            if track.mismatch >= 3 {
              track.lattice = fit
              track.mismatch = 0
            }
          } else {
            track.lattice = fit
            track.mismatch = 0
          }
        }
      } else {
        track.miss += 1
        if !cells.isEmpty { track.cells = cells }
        if track.miss > missMaximum || (cells.isEmpty && track.cells.isEmpty) {
          lost[id] = track
          tracks[id] = nil
        }
      }
    }

    // Paste only (not tracking): cells a track owned or held last frame and
    // nobody owns now stay with it, moved along, for up to `holdFrames`.
    var pasted = labels
    var nextShown: [Int: [(cell: Int, age: Int)]] = [:]
    for id in shown.keys.sorted() {
      let track = tracks[id] ?? lost[id]
      let vy = track?.vy ?? 0
      let vx = track?.vx ?? 0
      for entry in shown[id]! where entry.age < holdFrames {
        let y = entry.cell / gridWidth + vy
        let x = entry.cell % gridWidth + vx
        guard y >= 0, y < gridHeight, x >= 0, x < gridWidth else { continue }
        let target = y * gridWidth + x
        if pasted[target] == 0 {
          pasted[target] = Int32(id)
          nextShown[id, default: []].append((target, entry.age + 1))
        }
      }
    }
    for cell in 0..<cellCount where labels[cell] > 0 {
      nextShown[Int(labels[cell]), default: []].append((cell, 0))
    }
    shown = nextShown
    if tracks.isEmpty && lost.isEmpty { shown.removeAll() }
    let (paste, coverage) = pasteZones(pasted, support: support)
    return LabelFieldFrame(
      gridWidth: gridWidth, gridHeight: gridHeight, cellSize: Self.cellSize,
      paste: paste, coverage: coverage)
  }

  /// Each unowned cell near an owner goes to the nearest one: fully within
  /// the margin (1.5% of the crop's short side, at least 6 px) and fading
  /// with a smoothstep over 5% of it times the feather, like the
  /// per-detection mask-only paste. A cell the detector marks as mosaic is
  /// always fully covered.
  private func pasteZones(_ labels: [Int32], support: [Bool]) -> ([Int32], [Float]) {
    var margin: [Int32: Float] = [:]
    var fade: [Int32: Float] = [:]
    var bounds: [Int32: (top: Int, bottom: Int, left: Int, right: Int)] = [:]
    for cell in 0..<cellCount where labels[cell] > 0 {
      let y = cell / gridWidth
      let x = cell % gridWidth
      bounds[labels[cell]] = bounds[labels[cell]].map {
        (min($0.top, y), max($0.bottom, y), min($0.left, x), max($0.right, x))
      } ?? (y, y, x, x)
    }
    var reach: Float = 0
    for (label, bound) in bounds {
      let boxWidth = Float((bound.right - bound.left + 1) * Self.cellSize)
      let boxHeight = Float((bound.bottom - bound.top + 1) * Self.cellSize)
      let border = max(20, 0.06 * max(boxWidth, boxHeight))
      let shortSide = max(256, min(boxWidth, boxHeight) + 2 * border)
      margin[label] = max(6, shortSide * 0.015)
      fade[label] = feather > 0 ? max(3, shortSide * 0.05 * Float(feather)) : 0
      reach = max(reach, margin[label]! + fade[label]!)
    }
    // Chamfer distance (in pixels, cell centre to cell centre) to the
    // nearest owned cell, carrying that cell's owner.
    let step = Float(Self.cellSize)
    let diagonal = step * 1.414
    var distance = labels.map { $0 > 0 ? Float(0) : Float.infinity }
    var nearest = labels
    func relax(_ cell: Int, from other: Int, cost: Float) {
      let candidate = distance[other] + cost
      if candidate < distance[cell] && candidate <= reach + step {
        distance[cell] = candidate
        nearest[cell] = nearest[other]
      }
    }
    for y in 0..<gridHeight {
      for x in 0..<gridWidth {
        let cell = y * gridWidth + x
        if x > 0 { relax(cell, from: cell - 1, cost: step) }
        if y > 0 {
          relax(cell, from: cell - gridWidth, cost: step)
          if x > 0 { relax(cell, from: cell - gridWidth - 1, cost: diagonal) }
          if x + 1 < gridWidth { relax(cell, from: cell - gridWidth + 1, cost: diagonal) }
        }
      }
    }
    for y in stride(from: gridHeight - 1, through: 0, by: -1) {
      for x in stride(from: gridWidth - 1, through: 0, by: -1) {
        let cell = y * gridWidth + x
        if x + 1 < gridWidth { relax(cell, from: cell + 1, cost: step) }
        if y + 1 < gridHeight {
          relax(cell, from: cell + gridWidth, cost: step)
          if x + 1 < gridWidth { relax(cell, from: cell + gridWidth + 1, cost: diagonal) }
          if x > 0 { relax(cell, from: cell + gridWidth - 1, cost: diagonal) }
        }
      }
    }
    var paste = labels
    var coverage = labels.map { $0 > 0 ? Float(1) : 0 }
    for cell in 0..<cellCount where labels[cell] == 0 && nearest[cell] > 0 {
      let owner = nearest[cell]
      // The mask edge lies about half a cell from a cell centre.
      let gap = max(0, distance[cell] - step / 2)
      let ownMargin = margin[owner] ?? 6
      let ownFade = fade[owner] ?? 0
      let weight: Float
      if support[cell] || gap <= ownMargin {
        weight = 1
      } else if ownFade > 0 && gap < ownMargin + ownFade {
        let t = (gap - ownMargin) / ownFade
        weight = 1 - t * t * (3 - 2 * t)
      } else {
        weight = 0
      }
      if weight > 0 {
        paste[cell] = owner
        coverage[cell] = weight
      }
    }
    return (paste, coverage)
  }

  // MARK: - Orphans, audit, claim split, cleanup

  private func resolveOrphan(
    _ component: [Int], frame i: Int, labels: inout [Int32],
    edges: EdgeMaps?, cellScores: [Int: [Float]], confident: [Bool]
  ) {
    let bounds = cellBounds(component)
    let box = pixelBox(bounds)
    let fit = edges.flatMap { fitRegion(component, edges: $0) }
    var componentMask = [Bool](repeating: false, count: cellCount)
    for cell in component { componentMask[cell] = true }
    let ring = dilate(componentMask, radius: 2)
    var touchingCounts: [Int32: Int] = [:]
    for cell in 0..<cellCount where ring[cell] && labels[cell] > 0 {
      touchingCounts[labels[cell], default: 0] += 1
    }
    if let (touching, _) = touchingCounts.max(by: { lhs, rhs in
      lhs.value != rhs.value ? lhs.value < rhs.value : lhs.key > rhs.key
    }), let track = tracks[Int(touching)] {
      let confidentCells = component.filter { confident[$0] }
      var ownScore: Float?
      // A different grid must show on enough cells: a small piece of a
      // mosaic coming back from behind something has too few to judge.
      if let scores = cellScores[Int(touching)], confidentCells.count >= minimumCells {
        ownScore = median(confidentCells.map { scores[$0] })
      }
      // Only a measured, different grid makes it another mosaic (see the
      // novelty rule): a poorly explained piece alone is not evidence.
      let differs = (ownScore.map { $0 < 0.3 } ?? true) && differentGrid(fit, track.lattice)
      if !differs {
        for cell in component { labels[cell] = touching }
        return
      }
    }
    // A small fragment with no grid of its own joins the nearest label.
    if component.count < 150, fit == nil, labels.contains(where: { $0 > 0 }) {
      let nearest = nearestLabels(labels, maximumDistance: 16)
      if let cell = component.first(where: { nearest[$0] > 0 }) {
        let label = nearest[cell]
        for cell in component { labels[cell] = label }
        return
      }
    }
    var split: (lattices: [Lattice], labels: [Int], positions: [(Int, Int)]) = ([], [], [])
    if let edges, fit == nil || fit!.r < 0.65 {
      split = unsupervisedSplit(componentMask, bounds: bounds, edges: edges)
    }
    if split.lattices.count >= 2, let edges {
      var created: [Track] = []
      var seedsByTrack: [[Int]] = []
      for (index, lattice) in split.lattices.enumerated() {
        var cells: [Int] = []
        var marked = [Bool](repeating: false, count: cellCount)
        for (position, label) in zip(split.positions, split.labels) where label == index {
          for y in (position.1 + 16) / Self.cellSize..<(position.1 + 48) / Self.cellSize {
            for x in (position.0 + 16) / Self.cellSize..<(position.0 + 48) / Self.cellSize {
              let cell = y * gridWidth + x
              if y < gridHeight, x < gridWidth, componentMask[cell], !marked[cell] {
                marked[cell] = true
                cells.append(cell)
              }
            }
          }
        }
        guard !cells.isEmpty else { continue }
        var withConfidence = lattice
        withConfidence.r = 0.7
        let track = reidentify(box, lattice: lattice, frame: i)
          ?? newTrack(frame: i, lattice: withConfidence)
        created.append(track)
        seedsByTrack.append(cells)
      }
      guard !created.isEmpty else { return }
      let assigned = growInside(
        componentMask, seeds: seedsByTrack,
        lattices: created.map { $0.lattice }, edges: edges, confident: confident)
      for cell in component {
        let slot = assigned[cell]
        labels[cell] = Int32(created[slot > 0 ? Int(slot) - 1 : 0].id)
      }
      return
    }
    let track = reidentify(box, lattice: fit, frame: i) ?? newTrack(frame: i, lattice: fit)
    for cell in component { labels[cell] = Int32(track.id) }
  }

  private func differentGrid(_ first: Lattice?, _ second: Lattice?) -> Bool {
    guard let first, let second, first.r >= 0.7, second.r >= 0.7 else { return false }
    return abs(first.px - second.px) >= 1 || abs(first.py - second.py) >= 1
  }

  private func reidentify(_ box: LabelFieldBox, lattice: Lattice?, frame i: Int) -> Track? {
    var best: Track?
    var bestGap = Int.max
    for id in lost.keys.sorted() {
      guard let candidate = lost[id], i - candidate.last <= lostKeep,
        let last = candidate.lastBox
      else { continue }
      let gap = max(
        box.left - last.right, last.left - box.right,
        box.top - last.bottom, last.top - box.bottom)
      guard gap <= reidPixels else { continue }
      if let lattice, lattice.r >= 0.7, let old = candidate.lattice, old.r >= 0.7,
        abs(lattice.px - old.px) > 1.5 || abs(lattice.py - old.py) > 1.5
      {
        continue
      }
      if gap < bestGap {
        best = candidate
        bestGap = gap
      }
    }
    if let best {
      lost[best.id] = nil
      best.miss = 0
      tracks[best.id] = best
    }
    return best
  }

  private func newTrack(frame: Int, lattice: Lattice?) -> Track {
    let track = Track(id: nextID, frame: frame, lattice: lattice)
    nextID += 1
    tracks[track.id] = track
    return track
  }

  private func auditSplit(
    _ track: Track, frame i: Int, labels: inout [Int32],
    edges: EdgeMaps, confident: [Bool]
  ) {
    let cells = (0..<cellCount).filter { labels[$0] == Int32(track.id) }
    guard cells.count >= 600 else { return }
    guard let fit = fitRegion(cells, edges: edges), fit.r < 0.65 else {
      track.splitPending = 0
      return
    }
    var mask = [Bool](repeating: false, count: cellCount)
    for cell in cells { mask[cell] = true }
    let bounds = cellBounds(cells)
    let split = unsupervisedSplit(mask, bounds: bounds, edges: edges)
    let windows = split.labels.count
    let good = split.lattices.indices.filter { index in
      Double(split.labels.filter { $0 == index }.count)
        >= max(4, 0.15 * Double(windows))
    }
    guard good.count >= 2 else {
      track.splitPending = 0
      return
    }
    track.splitPending += 1
    guard track.splitPending >= 2 else { return }
    let keep: Int
    if let lattice = track.lattice {
      keep = good.min { lhs, rhs in
        abs(split.lattices[lhs].px - lattice.px) + abs(split.lattices[lhs].py - lattice.py)
          < abs(split.lattices[rhs].px - lattice.px) + abs(split.lattices[rhs].py - lattice.py)
      }!
    } else {
      keep = good.max { lhs, rhs in
        split.labels.filter { $0 == lhs }.count < split.labels.filter { $0 == rhs }.count
      }!
    }
    var members: [Track] = []
    var seeds: [[Int]] = []
    for index in good {
      var lattice = split.lattices[index]
      lattice.r = 0.7
      let member: Track
      if index == keep {
        track.lattice = lattice
        member = track
      } else {
        member = newTrack(frame: i, lattice: lattice)
      }
      var seedCells: [Int] = []
      for (position, label) in zip(split.positions, split.labels) where label == index {
        for y in (position.1 + 16) / Self.cellSize..<(position.1 + 48) / Self.cellSize {
          for x in (position.0 + 16) / Self.cellSize..<(position.0 + 48) / Self.cellSize {
            guard y < gridHeight, x < gridWidth else { continue }
            let cell = y * gridWidth + x
            if mask[cell] { seedCells.append(cell) }
          }
        }
      }
      members.append(member)
      seeds.append(seedCells)
    }
    let assigned = growInside(
      mask, seeds: seeds, lattices: members.map { $0.lattice },
      edges: edges, confident: confident)
    for cell in cells {
      let slot = assigned[cell]
      labels[cell] = Int32(slot > 0 ? members[Int(slot) - 1].id : track.id)
    }
    track.splitPending = 0
  }

  private func claimSplit(
    frame i: Int, labels: inout [Int32], observations: [LabelFieldObservation],
    claimsByTrack: [Int: [Int]], predicted: [Int: [Bool]]
  ) {
    for id in tracks.keys.sorted() {
      guard let track = tracks[id] else { continue }
      let label = Int32(id)
      var pair: (Int, Int)?
      let claims = claimsByTrack[id] ?? []
      if claims.count >= 2 {
        let insideCounts = claims.map { index in
          observations[index].cells.reduce(0) { $0 + (labels[$1] == label ? 1 : 0) }
        }
        search: for a in claims.indices {
          for b in claims.indices where b > a {
            let first = observations[claims[a]]
            let second = observations[claims[b]]
            guard first.cells.count >= 60, second.cells.count >= 60,
              Double(insideCounts[a]) >= 0.5 * Double(first.cells.count),
              Double(insideCounts[b]) >= 0.5 * Double(second.cells.count)
            else { continue }
            let firstSet = Set(first.cells)
            let shared = second.cells.reduce(0) { $0 + (firstSet.contains($1) ? 1 : 0) }
            if Double(shared) < 0.1 * Double(min(first.cells.count, second.cells.count)) {
              pair = (claims[a], claims[b])
              break search
            }
          }
        }
      }
      track.claimSplitHistory.append(pair != nil)
      if track.claimSplitHistory.count > 5 { track.claimSplitHistory.removeFirst() }
      guard let pair, track.claimSplitHistory.filter({ $0 }).count >= 3 else { continue }
      let region = (0..<cellCount).filter { labels[$0] == label }
      var mask = [Bool](repeating: false, count: cellCount)
      for cell in region { mask[cell] = true }
      let prior = predicted[id]
      func overlap(_ index: Int) -> Int {
        observations[index].cells.reduce(0) { $0 + ((prior?[$1] ?? false) ? 1 : 0) }
      }
      let (keepIndex, splitIndex) = overlap(pair.0) >= overlap(pair.1)
        ? (pair.0, pair.1) : (pair.1, pair.0)
      let keepCells = Set(observations[keepIndex].cells)
      let splitCells = Set(observations[splitIndex].cells)
      let seeds = [
        observations[keepIndex].cells.filter { mask[$0] && !splitCells.contains($0) },
        observations[splitIndex].cells.filter { mask[$0] && !keepCells.contains($0) },
      ]
      guard !seeds[0].isEmpty, !seeds[1].isEmpty else { continue }
      let newcomer = newTrack(frame: i, lattice: nil)
      let assigned = growInside(
        mask, seeds: seeds, lattices: [nil, nil], edges: nil, confident: [])
      for cell in region where assigned[cell] == 2 {
        labels[cell] = Int32(newcomer.id)
      }
      track.claimSplitHistory.removeAll()
    }
  }

  private func reattachDetachedPieces(
    labels: inout [Int32], cellScores: [Int: [Float]], confident: [Bool]
  ) {
    let present = Set(labels.filter { $0 > 0 }).sorted()
    for label in present {
      let mask = labels.map { $0 == label }
      let pieces = components(mask, eightConnected: true)
      let track = tracks[Int(label)]
      guard pieces.count > 1 else {
        track?.detached = 0
        continue
      }
      let total = pieces.reduce(0) { $0 + $1.count }
      let main = pieces.indices.max { pieces[$0].count < pieces[$1].count }!
      let candidates = pieces.indices.filter {
        $0 != main && Double(pieces[$0].count) <= 0.3 * Double(total)
      }
      if let track {
        track.detached = candidates.isEmpty ? 0 : track.detached + 1
        if track.detached < 8 { continue }
      }
      for index in candidates {
        let piece = pieces[index]
        var pieceMask = [Bool](repeating: false, count: cellCount)
        for cell in piece { pieceMask[cell] = true }
        let ring = dilate(pieceMask, radius: 1)
        var adjacent: [Int32: Int] = [:]
        for cell in 0..<cellCount
        where ring[cell] && !pieceMask[cell] && labels[cell] > 0 && labels[cell] != label {
          adjacent[labels[cell], default: 0] += 1
        }
        guard let (other, _) = adjacent.max(by: { lhs, rhs in
          lhs.value != rhs.value ? lhs.value < rhs.value : lhs.key > rhs.key
        }) else { continue }
        let confidentCells = piece.filter { confident[$0] }
        if let own = cellScores[Int(label)], let theirs = cellScores[Int(other)],
          confidentCells.count >= 8,
          median(confidentCells.map { own[$0] })
            > median(confidentCells.map { theirs[$0] }) + 0.3
        {
          continue
        }
        for cell in piece { labels[cell] = other }
      }
    }
  }

  // MARK: - Competitive growth

  private struct HeapItem {
    var cost: Float
    var cell: Int32
    var slot: Int32
  }

  private func grow(
    region: [Bool], seeds: [Int32], mismatches: [[Float]?],
    previous: [Int32], gates: [[Bool]], claims: [Int32]
  ) -> [Int32] {
    var labels = [Int32](repeating: 0, count: cellCount)
    var best = [Float](repeating: .infinity, count: cellCount)
    var heap: [HeapItem] = []
    func push(_ item: HeapItem) {
      heap.append(item)
      var child = heap.count - 1
      while child > 0 {
        let parent = (child - 1) / 2
        if heap[parent].cost <= heap[child].cost { break }
        heap.swapAt(parent, child)
        child = parent
      }
    }
    func pop() -> HeapItem {
      let top = heap[0]
      let last = heap.removeLast()
      if !heap.isEmpty {
        heap[0] = last
        var parent = 0
        while true {
          let left = parent * 2 + 1
          let right = left + 1
          var smallest = parent
          if left < heap.count, heap[left].cost < heap[smallest].cost { smallest = left }
          if right < heap.count, heap[right].cost < heap[smallest].cost { smallest = right }
          if smallest == parent { break }
          heap.swapAt(parent, smallest)
          parent = smallest
        }
      }
      return top
    }
    for cell in 0..<cellCount where seeds[cell] > 0 && region[cell] {
      best[cell] = 0
      push(HeapItem(cost: 0, cell: Int32(cell), slot: seeds[cell]))
    }
    while !heap.isEmpty {
      let item = pop()
      let cell = Int(item.cell)
      if labels[cell] != 0 { continue }
      labels[cell] = item.slot
      let slot = Int(item.slot) - 1
      let y = cell / gridWidth
      let x = cell % gridWidth
      for (dy, dx) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
        let ny = y + dy
        let nx = x + dx
        guard ny >= 0, ny < gridHeight, nx >= 0, nx < gridWidth else { continue }
        let next = ny * gridWidth + nx
        guard region[next], labels[next] == 0, gates[slot][next] else { continue }
        var cost: Float = 1
        if let mismatch = mismatches[slot] { cost += alpha * mismatch[next] }
        if previous[next] != 0 && previous[next] != item.slot { cost += beta }
        if claims[next] > 0 && claims[next] != item.slot { cost += claimCost }
        let total = item.cost + cost
        if total < best[next] {
          best[next] = total
          push(HeapItem(cost: total, cell: Int32(next), slot: item.slot))
        }
      }
    }
    return labels
  }

  /// Splits one region among seed sets (1-based slots in the result).
  private func growInside(
    _ region: [Bool], seeds: [[Int]], lattices: [Lattice?],
    edges: EdgeMaps?, confident: [Bool]
  ) -> [Int32] {
    var seedMap = [Int32](repeating: 0, count: cellCount)
    for (index, cells) in seeds.enumerated() {
      for cell in cells { seedMap[cell] = Int32(index + 1) }
    }
    var mismatches: [[Float]?] = Array(repeating: nil, count: seeds.count)
    if let edges, !confident.isEmpty {
      var known: [Int: Lattice] = [:]
      for (index, lattice) in lattices.enumerated() {
        if let lattice { known[index] = lattice }
      }
      if known.count == seeds.count, let scores = windowScores(edges, lattices: known) {
        var maps: [Int: [Float]] = [:]
        var best = [Float](repeating: -9, count: cellCount)
        for (index, windowValues) in scores.scores {
          var map = [Float](repeating: -9, count: cellCount)
          for cell in 0..<cellCount where region[cell] && confident[cell] {
            map[cell] = windowValues[cellWindowRow[cell / gridWidth] * windowColumns
              + cellWindowColumn[cell % gridWidth]]
            best[cell] = max(best[cell], map[cell])
          }
          maps[index] = map
        }
        for index in seeds.indices {
          guard let map = maps[index] else { continue }
          var mismatch = [Float](repeating: 0, count: cellCount)
          for cell in 0..<cellCount where region[cell] && confident[cell] && best[cell] > 0.6 {
            mismatch[cell] = max(0, best[cell] - map[cell])
          }
          mismatches[index] = mismatch
        }
      }
    }
    return grow(
      region: region, seeds: seedMap, mismatches: mismatches,
      previous: [Int32](repeating: 0, count: cellCount),
      gates: Array(repeating: region, count: seeds.count),
      claims: [Int32](repeating: 0, count: cellCount))
  }

  // MARK: - Grid (lattice) evidence

  final class EdgeMaps {
    let left: Int
    let top: Int
    let width: Int
    let height: Int
    /// |ΔY| (clipped at 40) across vertical / horizontal block boundaries,
    /// only where both pixels lie inside the eroded mosaic support.
    let gx: [Float]
    let gy: [Float]
    let cellCovered: [Bool]
    let gridWidth: Int

    init(left: Int, top: Int, width: Int, height: Int,
      gx: [Float], gy: [Float], cellCovered: [Bool], gridWidth: Int)
    {
      self.left = left
      self.top = top
      self.width = width
      self.height = height
      self.gx = gx
      self.gy = gy
      self.cellCovered = cellCovered
      self.gridWidth = gridWidth
    }

    func coversCell(_ cell: Int, gridWidth: Int) -> Bool { cellCovered[cell] }
  }

  private func edgeMaps(support: [Bool], luma: (LabelFieldBox) -> [UInt8]) -> EdgeMaps? {
    guard support.contains(true) else { return nil }
    var bounds = (top: gridHeight, bottom: -1, left: gridWidth, right: -1)
    for cell in 0..<cellCount where support[cell] {
      let y = cell / gridWidth
      let x = cell % gridWidth
      bounds = (min(bounds.top, y), max(bounds.bottom, y), min(bounds.left, x), max(bounds.right, x))
    }
    let region = LabelFieldBox(
      left: max(0, bounds.left * Self.cellSize - 64),
      top: max(0, bounds.top * Self.cellSize - 64),
      right: min(width - 1, (bounds.right + 1) * Self.cellSize - 1 + 64),
      bottom: min(height - 1, (bounds.bottom + 1) * Self.cellSize - 1 + 64))
    let regionWidth = region.width
    let regionHeight = region.height
    let pixels = luma(region)
    guard pixels.count == regionWidth * regionHeight else { return nil }
    // A pixel counts when every pixel within 3 px lies in the support
    // (7x7 erosion of the cell mask; outside the frame counts as inside).
    func cellRange(_ low: Int, _ high: Int, _ limit: Int) -> ClosedRange<Int> {
      max(0, low / Self.cellSize)...min(limit - 1, max(0, high) / Self.cellSize)
    }
    var rowInside = [Bool](repeating: false, count: regionHeight * gridWidth)
    for y in 0..<regionHeight {
      let absoluteY = region.top + y
      let rows = cellRange(max(0, absoluteY - 3), min(height - 1, absoluteY + 3), gridHeight)
      for cx in 0..<gridWidth {
        var inside = true
        for cy in rows where !support[cy * gridWidth + cx] {
          inside = false
          break
        }
        rowInside[y * gridWidth + cx] = inside
      }
    }
    var eroded = [Bool](repeating: false, count: regionWidth * regionHeight)
    for y in 0..<regionHeight {
      for x in 0..<regionWidth {
        let absoluteX = region.left + x
        let columns = cellRange(max(0, absoluteX - 3), min(width - 1, absoluteX + 3), gridWidth)
        var inside = true
        for cx in columns where !rowInside[y * gridWidth + cx] {
          inside = false
          break
        }
        eroded[y * regionWidth + x] = inside
      }
    }
    var gx = [Float](repeating: 0, count: regionWidth * regionHeight)
    var gy = [Float](repeating: 0, count: regionWidth * regionHeight)
    pixels.withUnsafeBufferPointer { luma in
      for y in 0..<regionHeight {
        let row = y * regionWidth
        for x in 0..<regionWidth - 1 where eroded[row + x] && eroded[row + x + 1] {
          gx[row + x] = min(40, abs(Float(luma[row + x + 1]) - Float(luma[row + x])))
        }
        if y + 1 < regionHeight {
          for x in 0..<regionWidth where eroded[row + x] && eroded[row + regionWidth + x] {
            gy[row + x] = min(40, abs(Float(luma[row + regionWidth + x]) - Float(luma[row + x])))
          }
        }
      }
    }
    var covered = [Bool](repeating: false, count: cellCount)
    for cy in (region.top / Self.cellSize)...min(gridHeight - 1, region.bottom / Self.cellSize) {
      for cx in (region.left / Self.cellSize)...min(gridWidth - 1, region.right / Self.cellSize) {
        covered[cy * gridWidth + cx] = true
      }
    }
    return EdgeMaps(
      left: region.left, top: region.top, width: regionWidth, height: regionHeight,
      gx: gx, gy: gy, cellCovered: covered, gridWidth: gridWidth)
  }

  private struct WindowScores {
    var confident: [Bool]
    var scores: [Int: [Float]]
  }

  /// 64 px windows on a 16 px stride: edge energy and, per lattice, the
  /// phase-aligned concentration of that energy (x plus y, -2...2).
  private func windowScores(_ edges: EdgeMaps, lattices: [Int: Lattice]) -> WindowScores? {
    let rows = windowRows
    let columns = windowColumns
    let size = windowSize
    let stride = windowStride
    let firstRow = max(0, (edges.top - size) / stride)
    let lastRow = min(rows - 1, (edges.top + edges.height) / stride)
    let firstColumn = max(0, (edges.left - size) / stride)
    let lastColumn = min(columns - 1, (edges.left + edges.width) / stride)
    guard firstRow <= lastRow, firstColumn <= lastColumn else { return nil }
    let w = edges.width
    let h = edges.height
    // Column sums of gx over each window row, row sums of gy over each
    // window column (inside the region; outside the edges are zero).
    var columnPrefix = [Double](repeating: 0, count: (h + 1) * w)
    for y in 0..<h {
      for x in 0..<w {
        columnPrefix[(y + 1) * w + x] = columnPrefix[y * w + x] + Double(edges.gx[y * w + x])
      }
    }
    var rowPrefix = [Double](repeating: 0, count: h * (w + 1))
    for y in 0..<h {
      for x in 0..<w {
        rowPrefix[y * (w + 1) + x + 1] = rowPrefix[y * (w + 1) + x] + Double(edges.gy[y * w + x])
      }
    }
    var ex = [Double](repeating: 0, count: rows * columns)
    var ey = [Double](repeating: 0, count: rows * columns)
    var sx: [Int: [Double]] = [:]
    var sy: [Int: [Double]] = [:]
    for id in lattices.keys {
      sx[id] = [Double](repeating: 0, count: rows * columns)
      sy[id] = [Double](repeating: 0, count: rows * columns)
    }
    let ids = lattices.keys.sorted()
    // x: per window row, prefix over x of the column sums (times the phase).
    for row in firstRow...lastRow {
      let y0 = max(0, row * stride - edges.top)
      let y1 = min(h, row * stride + size - edges.top)
      guard y0 < y1 else { continue }
      var plain = [Double](repeating: 0, count: w + 1)
      var sums = [Double](repeating: 0, count: w)
      for x in 0..<w {
        sums[x] = columnPrefix[y1 * w + x] - columnPrefix[y0 * w + x]
        plain[x + 1] = plain[x] + sums[x]
      }
      var phased: [Int: (re: [Double], im: [Double])] = [:]
      for id in ids {
        let lattice = lattices[id]!
        var re = [Double](repeating: 0, count: w + 1)
        var im = [Double](repeating: 0, count: w + 1)
        for x in 0..<w {
          let angle = 2 * Double.pi * (Double(edges.left + x) + 1 - lattice.phx) / lattice.px
          re[x + 1] = re[x] + sums[x] * cos(angle)
          im[x + 1] = im[x] + sums[x] * sin(angle)
        }
        phased[id] = (re, im)
      }
      for column in firstColumn...lastColumn {
        let x0 = max(0, column * stride - edges.left)
        let x1 = min(w, column * stride + size - edges.left)
        guard x0 < x1 else { continue }
        let window = row * columns + column
        let energy = plain[x1] - plain[x0]
        ex[window] = energy
        for id in ids {
          let values = phased[id]!
          sx[id]![window] = (values.re[x1] - values.re[x0]) / (energy + 1e-6)
        }
      }
    }
    // y: per window column, prefix over y of the row sums.
    for column in firstColumn...lastColumn {
      let x0 = max(0, column * stride - edges.left)
      let x1 = min(w, column * stride + size - edges.left)
      guard x0 < x1 else { continue }
      var plain = [Double](repeating: 0, count: h + 1)
      var sums = [Double](repeating: 0, count: h)
      for y in 0..<h {
        sums[y] = rowPrefix[y * (w + 1) + x1] - rowPrefix[y * (w + 1) + x0]
        plain[y + 1] = plain[y] + sums[y]
      }
      var phased: [Int: [Double]] = [:]
      for id in ids {
        let lattice = lattices[id]!
        var re = [Double](repeating: 0, count: h + 1)
        for y in 0..<h {
          let angle = 2 * Double.pi * (Double(edges.top + y) + 1 - lattice.phy) / lattice.py
          re[y + 1] = re[y] + sums[y] * cos(angle)
        }
        phased[id] = re
      }
      for row in firstRow...lastRow {
        let y0 = max(0, row * stride - edges.top)
        let y1 = min(h, row * stride + size - edges.top)
        guard y0 < y1 else { continue }
        let window = row * columns + column
        let energy = plain[y1] - plain[y0]
        ey[window] = energy
        for id in ids {
          sy[id]![window] = (phased[id]![y1] - phased[id]![y0]) / (energy + 1e-6)
        }
      }
    }
    var confident = [Bool](repeating: false, count: rows * columns)
    for window in 0..<(rows * columns) {
      confident[window] = ex[window] > 400 && ey[window] > 400
    }
    var scores: [Int: [Float]] = [:]
    for id in ids {
      scores[id] = (0..<(rows * columns)).map { Float(sx[id]![$0] + sy[id]![$0]) }
    }
    return WindowScores(confident: confident, scores: scores)
  }

  private static let periods: [Double] = stride(from: 14.0, to: 48.0, by: 0.05).map { $0 }
  private static let coarsePeriods: [Double] = stride(from: 14.0, to: 48.0, by: 0.2).map { $0 }

  /// Fundamental grid period of an edge profile: among local maxima of the
  /// Rayleigh concentration within 90% of the best, the largest period
  /// (rejects P/2, P/3).
  private func fitAxis(
    _ profile: [Double], start: Double, periods: [Double]
  ) -> (p: Double, phase: Double, r: Double)? {
    let total = profile.reduce(0, +)
    guard total > 0 else { return nil }
    var concentration = [Double](repeating: 0, count: periods.count)
    var phases = [Double](repeating: 0, count: periods.count)
    for (index, period) in periods.enumerated() {
      let step = 2 * Double.pi / period
      var re = 0.0
      var im = 0.0
      // Rotate by one step per sample instead of calling cos/sin each time.
      var cr = cos(step * start)
      var ci = sin(step * start)
      let sr = cos(step)
      let si = sin(step)
      for value in profile {
        if value != 0 {
          re += value * cr
          im += value * ci
        }
        let nr = cr * sr - ci * si
        ci = cr * si + ci * sr
        cr = nr
      }
      concentration[index] = (re * re + im * im).squareRoot() / total
      var phase = atan2(im, re) / (2 * Double.pi) * period
      phase = phase.truncatingRemainder(dividingBy: period)
      if phase < 0 { phase += period }
      phases[index] = phase
    }
    guard let peak = concentration.max(), peak > 0 else { return nil }
    var maxima: [Int] = []
    for index in 1..<(periods.count - 1)
    where concentration[index] >= concentration[index - 1]
      && concentration[index] >= concentration[index + 1]
    {
      maxima.append(index)
    }
    if maxima.isEmpty { maxima = [concentration.firstIndex(of: peak)!] }
    let best = maxima.map { concentration[$0] }.max()!
    let chosen = maxima.filter { concentration[$0] >= 0.9 * best }.max()!
    return (periods[chosen], phases[chosen], concentration[chosen])
  }

  private func phaseScore(_ profile: [Double], start: Double, period: Double, phase: Double) -> Double {
    let total = profile.reduce(0, +)
    guard total > 0 else { return 0 }
    var re = 0.0
    for (index, value) in profile.enumerated() where value != 0 {
      re += value * cos(2 * Double.pi * (start + Double(index) - phase) / period)
    }
    return re / total
  }

  /// Grid of a cell region (x and y), from its own eroded pixels.
  private func fitRegion(_ cells: [Int], edges: EdgeMaps) -> Lattice? {
    guard cells.count >= 190 else { return nil }
    let bounds = cellBounds(cells)
    let left = bounds.left * Self.cellSize
    let top = bounds.top * Self.cellSize
    let right = min(width, (bounds.right + 1) * Self.cellSize)
    let bottom = min(height, (bounds.bottom + 1) * Self.cellSize)
    let w = right - left
    let h = bottom - top
    guard w > 1, h > 1 else { return nil }
    var cellMask = [Bool](repeating: false, count: cellCount)
    for cell in cells { cellMask[cell] = true }
    // 5x5 erosion of the upsampled cell mask (outside the box counts as in).
    func inMask(_ x: Int, _ y: Int) -> Bool {
      if x < left || x >= right || y < top || y >= bottom { return true }
      return cellMask[(y / Self.cellSize) * gridWidth + x / Self.cellSize]
    }
    var mask = [Bool](repeating: false, count: w * h)
    var kept = 0
    for y in 0..<h {
      for x in 0..<w {
        var inside = true
        outer: for dy in -2...2 {
          for dx in -2...2 where !inMask(left + x + dx, top + y + dy) {
            inside = false
            break outer
          }
        }
        mask[y * w + x] = inside
        if inside { kept += 1 }
      }
    }
    guard kept >= 2500 else { return nil }
    var profileX = [Double](repeating: 0, count: w)
    var profileY = [Double](repeating: 0, count: h)
    for y in 0..<h {
      let edgeY = top + y - edges.top
      guard edgeY >= 0, edgeY < edges.height else { continue }
      for x in 0..<w {
        let edgeX = left + x - edges.left
        guard edgeX >= 0, edgeX < edges.width else { continue }
        let index = edgeY * edges.width + edgeX
        if mask[y * w + x] && (x + 1 < w ? mask[y * w + x + 1] : false) {
          profileX[x] += Double(edges.gx[index])
        }
        if mask[y * w + x] && (y + 1 < h ? mask[(y + 1) * w + x] : false) {
          profileY[y] += Double(edges.gy[index])
        }
      }
    }
    guard let fx = fitAxis(profileX, start: Double(left) + 1, periods: Self.periods),
      let fy = fitAxis(profileY, start: Double(top) + 1, periods: Self.periods)
    else { return nil }
    return Lattice(px: fx.p, phx: fx.phase, py: fy.p, phy: fy.phase, r: min(fx.r, fy.r))
  }

  /// RANSAC split of a region into grids, from 64 px windows on a 32 px
  /// stride (evidence/an_unsup.py). Returns the grids, each window's grid
  /// (-1 = none) and the windows' top-left pixels.
  private func unsupervisedSplit(
    _ mask: [Bool], bounds: (top: Int, bottom: Int, left: Int, right: Int), edges: EdgeMaps
  ) -> (lattices: [Lattice], labels: [Int], positions: [(Int, Int)]) {
    let left = bounds.left * Self.cellSize
    let top = bounds.top * Self.cellSize
    let right = min(width, (bounds.right + 1) * Self.cellSize)
    let bottom = min(height, (bounds.bottom + 1) * Self.cellSize)
    var windows: [(px: [Double], py: [Double], x: Int, y: Int)] = []
    var y = top
    while y + 64 <= bottom {
      var x = left
      while x + 64 <= right {
        var inside = 0
        for cy in (y / Self.cellSize)..<((y + 64) / Self.cellSize) {
          for cx in (x / Self.cellSize)..<((x + 64) / Self.cellSize)
          where mask[cy * gridWidth + cx] {
            inside += 1
          }
        }
        if Double(inside) >= 0.85 * 256 {
          var px = [Double](repeating: 0, count: 64)
          var py = [Double](repeating: 0, count: 64)
          for dy in 0..<64 {
            let edgeY = y + dy - edges.top
            guard edgeY >= 0, edgeY < edges.height else { continue }
            for dx in 0..<64 {
              let edgeX = x + dx - edges.left
              guard edgeX >= 0, edgeX < edges.width else { continue }
              let index = edgeY * edges.width + edgeX
              px[dx] += Double(edges.gx[index])
              py[dy] += Double(edges.gy[index])
            }
          }
          windows.append((px, py, x, y))
        }
        x += 32
      }
      y += 32
    }
    let count = windows.count
    let positions = windows.map { ($0.x, $0.y) }
    guard count >= 4 else { return ([], [], positions) }
    func score(_ index: Int, _ lattice: Lattice) -> (Double, Double) {
      let window = windows[index]
      return (
        phaseScore(window.px, start: Double(window.x) + 1, period: lattice.px, phase: lattice.phx),
        phaseScore(window.py, start: Double(window.y) + 1, period: lattice.py, phase: lattice.phy))
    }
    var hypotheses: [Lattice] = []
    for window in windows {
      if let fx = fitAxis(window.px, start: Double(window.x) + 1, periods: Self.coarsePeriods),
        let fy = fitAxis(window.py, start: Double(window.y) + 1, periods: Self.coarsePeriods),
        fx.r >= 0.5, fy.r >= 0.5
      {
        hypotheses.append(Lattice(px: fx.p, phx: fx.phase, py: fy.p, phy: fy.phase, r: min(fx.r, fy.r)))
      }
    }
    func same(_ a: Lattice, _ b: Lattice) -> Bool {
      func phaseDistance(_ period: Double, _ first: Double, _ second: Double) -> Double {
        var distance = (first - second).truncatingRemainder(dividingBy: period)
        if distance < 0 { distance += period }
        return min(distance, period - distance)
      }
      return abs(a.px - b.px) < 0.3 && abs(a.py - b.py) < 0.3
        && phaseDistance(a.px, a.phx, b.phx) < 1.5 && phaseDistance(a.py, a.phy, b.phy) < 1.5
    }
    func support(_ lattice: Lattice, _ indices: Set<Int>) -> [Int] {
      indices.sorted().filter { index in
        let (a, b) = score(index, lattice)
        return a >= 0.45 && b >= 0.45 && a + b >= 1.2
      }
    }
    var remaining = Set(0..<count)
    var lattices: [Lattice] = []
    while !remaining.isEmpty && !hypotheses.isEmpty {
      var best: Lattice?
      var bestSupport: [Int] = []
      for hypothesis in hypotheses.prefix(60) where !lattices.contains(where: { same(hypothesis, $0) }) {
        let supported = support(hypothesis, remaining)
        if supported.count > bestSupport.count {
          best = hypothesis
          bestSupport = supported
        }
      }
      guard let chosen = best,
        Double(bestSupport.count) >= max(4, 0.1 * Double(count))
      else { break }
      // Refine from the summed profiles of the supporting windows.
      var profileX = [Double](repeating: 0, count: right - left + 2)
      var profileY = [Double](repeating: 0, count: bottom - top + 2)
      for index in bestSupport {
        let window = windows[index]
        for dx in 0..<64 { profileX[window.x - left + dx] += window.px[dx] }
        for dy in 0..<64 { profileY[window.y - top + dy] += window.py[dy] }
      }
      var lattice = chosen
      if let fx = fitAxis(profileX, start: Double(left) + 1, periods: Self.periods),
        let fy = fitAxis(profileY, start: Double(top) + 1, periods: Self.periods)
      {
        let refined = Lattice(px: fx.p, phx: fx.phase, py: fy.p, phy: fy.phase, r: min(fx.r, fy.r))
        if !lattices.contains(where: { same(refined, $0) }) { lattice = refined }
      }
      lattices.append(lattice)
      remaining.subtract(support(lattice, remaining))
      remaining.subtract(bestSupport)
    }
    let labels = (0..<count).map { index -> Int in
      let totals = lattices.map { lattice -> Double in
        let (a, b) = score(index, lattice)
        return a + b
      }
      guard let best = totals.max(), best >= 0.8 else { return -1 }
      return totals.firstIndex(of: best)!
    }
    return (lattices, labels, positions)
  }

  // MARK: - Grid helpers

  private func coverObservations(_ observations: [LabelFieldObservation]) -> Set<Int> {
    var covers = Set<Int>()
    for (index, observation) in observations.enumerated() {
      let inner = observations.indices.filter { other in
        other != index && observations[other].box.area < observation.box.area
          && Double(observations[other].box.intersectionArea(observation.box))
            >= 0.7 * Double(observations[other].box.area)
      }
      var disjoint = false
      search: for a in inner.indices {
        for b in inner.indices where b > a {
          let first = observations[inner[a]].box
          let second = observations[inner[b]].box
          if Double(first.intersectionArea(second))
            < 0.1 * Double(min(first.area, second.area))
          {
            disjoint = true
            break search
          }
        }
      }
      if disjoint { covers.insert(index) }
    }
    return covers
  }

  private func cellBounds(_ cells: [Int]) -> (top: Int, bottom: Int, left: Int, right: Int) {
    var bounds = (top: Int.max, bottom: Int.min, left: Int.max, right: Int.min)
    for cell in cells {
      let y = cell / gridWidth
      let x = cell % gridWidth
      bounds = (min(bounds.top, y), max(bounds.bottom, y), min(bounds.left, x), max(bounds.right, x))
    }
    return bounds
  }

  private func pixelBox(_ bounds: (top: Int, bottom: Int, left: Int, right: Int)) -> LabelFieldBox {
    LabelFieldBox(
      left: bounds.left * Self.cellSize,
      top: bounds.top * Self.cellSize,
      right: min(width - 1, (bounds.right + 1) * Self.cellSize - 1),
      bottom: min(height - 1, (bounds.bottom + 1) * Self.cellSize - 1))
  }

  private func dilate(_ mask: [Bool], radius: Int) -> [Bool] {
    var horizontal = [Bool](repeating: false, count: cellCount)
    for y in 0..<gridHeight {
      let row = y * gridWidth
      for x in 0..<gridWidth where mask[row + x] {
        for nx in max(0, x - radius)...min(gridWidth - 1, x + radius) {
          horizontal[row + nx] = true
        }
      }
    }
    var result = [Bool](repeating: false, count: cellCount)
    for y in 0..<gridHeight {
      for x in 0..<gridWidth where horizontal[y * gridWidth + x] {
        for ny in max(0, y - radius)...min(gridHeight - 1, y + radius) {
          result[ny * gridWidth + x] = true
        }
      }
    }
    return result
  }

  private func erode(_ mask: [Bool], radius: Int) -> [Bool] {
    let inverted = mask.map { !$0 }
    return dilate(inverted, radius: radius).map { !$0 }
  }

  private func components(_ mask: [Bool], eightConnected: Bool) -> [[Int]] {
    var seen = [Bool](repeating: false, count: cellCount)
    var result: [[Int]] = []
    let offsets = eightConnected
      ? [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
      : [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for start in 0..<cellCount where mask[start] && !seen[start] {
      var component: [Int] = [start]
      seen[start] = true
      var head = 0
      while head < component.count {
        let cell = component[head]
        head += 1
        let y = cell / gridWidth
        let x = cell % gridWidth
        for (dy, dx) in offsets {
          let ny = y + dy
          let nx = x + dx
          guard ny >= 0, ny < gridHeight, nx >= 0, nx < gridWidth else { continue }
          let next = ny * gridWidth + nx
          if mask[next] && !seen[next] {
            seen[next] = true
            component.append(next)
          }
        }
      }
      result.append(component)
    }
    return result
  }

  /// Nearest label (8-connected steps) for unlabeled cells within a distance.
  private func nearestLabels(_ labels: [Int32], maximumDistance: Int) -> [Int32] {
    var result = labels
    var distance = [Int](repeating: Int.max, count: cellCount)
    var frontier: [Int] = []
    for cell in 0..<cellCount where labels[cell] != 0 {
      distance[cell] = 0
      frontier.append(cell)
    }
    var step = 0
    while !frontier.isEmpty && step < maximumDistance {
      step += 1
      var next: [Int] = []
      for cell in frontier {
        let y = cell / gridWidth
        let x = cell % gridWidth
        for dy in -1...1 {
          for dx in -1...1 where dy != 0 || dx != 0 {
            let ny = y + dy
            let nx = x + dx
            guard ny >= 0, ny < gridHeight, nx >= 0, nx < gridWidth else { continue }
            let neighbour = ny * gridWidth + nx
            if distance[neighbour] == Int.max {
              distance[neighbour] = step
              result[neighbour] = result[cell]
              next.append(neighbour)
            }
          }
        }
      }
      frontier = next
    }
    return result
  }

  private func median(_ values: [Float]) -> Float {
    guard !values.isEmpty else { return 0 }
    let sorted = values.sorted()
    let middle = sorted.count / 2
    return sorted.count % 2 == 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2
  }
}
