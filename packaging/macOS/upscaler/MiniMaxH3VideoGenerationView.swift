import AppKit
import AVFoundation
import Foundation
import SwiftUI
import UniformTypeIdentifiers

fileprivate enum MiniMaxH3MusicVideoContinuationMode: String, CaseIterable {
  case hybridAV = "hybrid-av"
  case latentPrefix = "latent-prefix"
  case firstFrame = "first"
  case firstAndProvidedLast = "first-last-provided"
  case firstAndGeneratedLast = "first-last-generated"
}

fileprivate enum MiniMaxH3AudioConditioningMode: String, CaseIterable {
  case backgroundMusic = "background-music"
  case lipSync = "lip-sync"

  var label: String {
    switch self {
    case .backgroundMusic:
      return "BGM参照（口パクしない）"
    case .lipSync:
      return "リップシンク"
    }
  }

  var helpText: String {
    switch self {
    case .backgroundMusic:
      return "音源はテンポ・曲構成・雰囲気の参照と完成動画のBGMに使います。人物の口は歌詞や声に同期させません。"
    case .lipSync:
      return "音源を声・歌唱のタイミング条件として使い、人物の口や表情を音声に合わせます。"
    }
  }
}

private struct MiniMaxH3UIProgressEvent: Decodable {
  let stage: String
  let state: String
  let progress: Double
  let message: String
}

private struct MiniMaxH3ResolutionProfile: Identifiable, Hashable {
  let width: Int
  let height: Int
  let outputWidth: Int
  let outputHeight: Int
  let note: String
  let fixedDuration: Double?

  init(
    width: Int,
    height: Int,
    outputWidth: Int? = nil,
    outputHeight: Int? = nil,
    note: String,
    fixedDuration: Double? = nil
  ) {
    self.width = width
    self.height = height
    self.outputWidth = outputWidth ?? width
    self.outputHeight = outputHeight ?? height
    self.note = note
    self.fixedDuration = fixedDuration
  }

  var id: String { "\(outputWidth)x\(outputHeight)" }
  var label: String { "\(outputWidth)×\(outputHeight)（\(note)）" }

  static let supported: [Self] = [
    .init(width: 640, height: 352, note: "低負荷・横"),
    .init(width: 864, height: 480, note: "標準・横"),
    .init(width: 960, height: 544, note: "高精細・横"),
    .init(width: 1024, height: 576, note: "高精細・横"),
    .init(width: 1344, height: 768, note: "H3-Base上限・横"),
    // The VAE/DiT canvas must be divisible by 32. Generate eight extra rows,
    // then center-crop them at the AVFoundation writer boundary so the file is
    // exact Full HD rather than a stretched or scaled approximation.
    .init(
      width: 1920,
      height: 1088,
      outputWidth: 1920,
      outputHeight: 1080,
      note: "本家1080p・6秒",
      fixedDuration: 6
    ),
    .init(width: 640, height: 640, note: "正方形"),
    .init(width: 768, height: 768, note: "最高精細・正方形"),
    .init(width: 480, height: 864, note: "標準・縦"),
    .init(width: 576, height: 1024, note: "高精細・縦"),
    .init(width: 768, height: 1344, note: "H3-Base上限・縦"),
  ]
}

fileprivate struct MiniMaxH3MusicCutPoint: Identifiable, Equatable {
  let id: UUID
  var seconds: Double

  init(id: UUID = UUID(), seconds: Double) {
    self.id = id
    self.seconds = seconds
  }
}

enum MiniMaxH3AIPromptProvider: String, CaseIterable, Identifiable {
  case openAICompatible = "openai-compatible"
  case ollama = "ollama"
  case lmStudio = "lm-studio"
  case custom = "custom"

  var id: String { rawValue }

  var label: String {
    switch self {
    case .openAICompatible:
      return "OpenAI互換 / Gemma"
    case .ollama:
      return "Ollama"
    case .lmStudio:
      return "LM Studio"
    case .custom:
      return "カスタム"
    }
  }

  var defaultBaseURL: String {
    switch self {
    case .openAICompatible:
      return "http://127.0.0.1:18080/v1"
    case .ollama:
      return "http://127.0.0.1:11434/v1"
    case .lmStudio:
      return "http://127.0.0.1:1234/v1"
    case .custom:
      return "http://127.0.0.1:18080/v1"
    }
  }
}

private struct MiniMaxH3AIChatRequest: Encodable {
  struct Message: Encodable {
    let role: String
    let content: String
  }

  let model: String
  let temperature: Double
  let maxTokens: Int
  let messages: [Message]

  private enum CodingKeys: String, CodingKey {
    case model
    case temperature
    case maxTokens = "max_tokens"
    case messages
  }
}

private struct MiniMaxH3AIModelsResponse: Decodable {
  struct Model: Decodable {
    let id: String?
    let model: String?
    let name: String?
  }

  let data: [Model]?
  let models: [Model]?
}

private struct MiniMaxH3AIChatResponse: Decodable {
  struct Choice: Decodable {
    struct Message: Decodable {
      let content: String?
    }

    let text: String?
    let message: Message?
  }

  struct Message: Decodable {
    let content: String?
  }

  let choices: [Choice]?
  let message: Message?
  let response: String?
}

@MainActor
final class MiniMaxH3Controller: ObservableObject {
  private static let maximumIdentityImages = 8
  private static let maximumAudioConditioningDuration = 10.0
  private static let maximumVisibleLogCharacters = 24_000
  private static let retainedLogCharacters = 16_000
  private static let manifestPathDefaultsKey =
    "com.okatti.mioh.upscaler.10erosMaxH3ManifestPath"
  private static let legacyManifestPathDefaultsKey =
    "com.okatti.lada.coreai.10erosMaxH3ManifestPath"
  private static let coreAICacheRootDefaultsKey =
    "com.okatti.mioh.upscaler.h3CoreAICacheRoot"

  @Published var prompt = "モザイクを除去して最高品質の動画を生成する。"
  @Published var aiPromptProvider: MiniMaxH3AIPromptProvider = .openAICompatible
  @Published var aiPromptRequest = ""
  @Published var aiGeneratedPrompt = ""
  @Published var aiPromptAPIURL = "http://127.0.0.1:18080/v1"
  @Published var aiPromptAPIKey = ""
  @Published var aiPromptModel = "auto"
  @Published private(set) var isGeneratingAIPrompt = false
  @Published var backend = "coreai"
  @Published var resolutionProfileID = "864x480" {
    didSet {
      guard let profile = Self.resolutionProfiles.first(where: {
        $0.id == resolutionProfileID
      }) else { return }
      width = profile.width
      height = profile.height
      outputWidth = profile.outputWidth
      outputHeight = profile.outputHeight
      if let fixedDuration = profile.fixedDuration {
        duration = fixedDuration
      }
    }
  }
  @Published private(set) var width = 864
  @Published private(set) var height = 480
  @Published private(set) var outputWidth = 864
  @Published private(set) var outputHeight = 480
  @Published var duration = 10.0
  @Published var seed = "261662374822964"
  @Published var manifestPath: String {
    didSet {
      UserDefaults.standard.set(
        manifestPath,
        forKey: Self.manifestPathDefaultsKey
      )
      refreshConditioningMode()
    }
  }
  @Published var coreAICacheRoot: String {
    didSet {
      UserDefaults.standard.set(
        coreAICacheRoot,
        forKey: Self.coreAICacheRootDefaultsKey
      )
      refreshCoreAICacheSummary()
    }
  }
  @Published private(set) var coreAICacheSummary = "システム標準"
  @Published private(set) var supportsPromptOnly = false
  @Published private(set) var inputURLs: [URL] = []
  @Published private(set) var audioInputURL: URL?
  @Published fileprivate var audioConditioningMode:
    MiniMaxH3AudioConditioningMode =
    .backgroundMusic
  @Published var audioStartSeconds = 0.0
  @Published private(set) var audioDurationSeconds: Double?
  @Published var lyricsText = ""
  @Published var lyricsSearchQuery = ""
  @Published var musicVideoMode = false
  @Published fileprivate var musicVideoContinuationMode:
    MiniMaxH3MusicVideoContinuationMode =
    .hybridAV
  @Published var musicVideoLastFrameDirectory = ""
  @Published var usesAutomaticMusicCuts = true
  @Published fileprivate var musicCutPoints: [MiniMaxH3MusicCutPoint] = []
  @Published private(set) var selectedMusicCutPointID: UUID?
  @Published private(set) var musicPreviewSeconds: Double?
  @Published private(set) var usesUpscalerInput = true
  @Published var imageReferenceScope = MiniMaxH3ImageReferenceScope.wholeImage
  @Published private(set) var faceReferences: [MiniMaxH3FaceReference] = []
  @Published private(set) var isDetectingFaces = false
  @Published var outputPath = ""
  @Published var progress = 0.0
  @Published var status = "待機中"
  @Published var log = ""
  @Published private(set) var musicAnalysisSummary = ""
  @Published var isRunning = false

  private var process: Process?
  private var standardOutputBuffer = Data()
  private var standardErrorBuffer = Data()
  private var currentUpscalerInput: URL?
  private var automaticOutputPath = ""
  private var faceDetectionTask: Task<Void, Never>?
  private var faceReferenceDirectory: URL?
  private var audioPreviewPlayer: AVPlayer?
  private var audioPreviewURL: URL?
  private var audioPreviewStopTask: Task<Void, Never>?
  private var lastAudioPreviewSeconds: Double?

  fileprivate static let resolutionProfiles = MiniMaxH3ResolutionProfile.supported

  private var selectedResolutionProfile: MiniMaxH3ResolutionProfile? {
    Self.resolutionProfiles.first { $0.id == resolutionProfileID }
  }

  var isDurationFixed: Bool {
    selectedResolutionProfile?.fixedDuration != nil
  }

  var maximumShotDuration: Double {
    audioInputURL == nil ? 15 : Self.maximumAudioConditioningDuration
  }

  init() {
    let savedManifestPath = UserDefaults.standard.string(
      forKey: Self.manifestPathDefaultsKey
    ) ?? UserDefaults.standard.string(
      forKey: Self.legacyManifestPathDefaultsKey
    ) ?? ""
    manifestPath = Self.resolvePipelineManifestPath(savedManifestPath)
    coreAICacheRoot = UserDefaults.standard.string(
      forKey: Self.coreAICacheRootDefaultsKey
    ) ?? ""
    refreshConditioningMode()
    refreshCoreAICacheSummary()
  }

  var supportsRuntime: Bool {
    ProcessInfo.processInfo.operatingSystemVersion.majorVersion >= 27
  }

  var runnerURL: URL? {
    Bundle.main.resourceURL?.appendingPathComponent(
      "bin/mioh-minimax-h3-native"
    )
  }

  var modelReady: Bool {
    Self.isPipelineManifest(
      URL(fileURLWithPath: effectiveManifestPath)
    )
  }

  private var effectiveManifestPath: String {
    Self.resolvePipelineManifestPath(manifestPath)
  }

  var isImageSequence: Bool {
    !inputURLs.isEmpty && inputURLs.allSatisfy(Self.isImage)
  }

  var inputSummary: String {
    if supportsPromptOnly { return "なし（プロンプトのみ）" }
    if inputURLs.isEmpty { return "未指定" }
    if isImageSequence {
      if imageReferenceScope == .faceOnly {
        if isDetectingFaces { return "Subject候補の顔を検出中" }
        return "顔参照 \(selectedFaceReferences.count)件（検出\(faceReferences.count)件）"
      }
      if inputURLs.count == 1 { return inputURLs[0].path }
      return "Subject参照画像 \(inputURLs.count)枚（先頭画像が基準）"
    }
    return inputURLs[0].path
  }

  func canStart() -> Bool {
    guard !isRunning, supportsRuntime, modelReady, validInputSelection,
      !outputPath.isEmpty,
      !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
      duration.isFinite, duration >= 2, duration <= 15,
      audioStartSeconds.isFinite, audioStartSeconds >= 0,
      UInt64(seed) != nil,
      let runnerURL,
      FileManager.default.isExecutableFile(atPath: runnerURL.path)
    else { return false }
    if audioInputURL != nil {
      guard isImageSequence, duration <= Self.maximumAudioConditioningDuration
      else { return false }
      guard let audioDurationSeconds,
        audioDurationSeconds - audioStartSeconds > 0
      else { return false }
      if musicVideoMode {
        guard audioDurationSeconds - audioStartSeconds >= 2 else {
          return false
        }
        if musicVideoContinuationMode == .firstAndProvidedLast {
          var isDirectory: ObjCBool = false
          guard !musicVideoLastFrameDirectory.isEmpty,
            FileManager.default.fileExists(
              atPath: musicVideoLastFrameDirectory,
              isDirectory: &isDirectory
            ),
            isDirectory.boolValue
          else { return false }
        }
        if !usesAutomaticMusicCuts {
          let boundaries = [0.0] + musicCutPoints.map(\.seconds).sorted()
            + [audioDurationSeconds - audioStartSeconds]
          return zip(boundaries, boundaries.dropFirst()).allSatisfy { pair in
            pair.1 - pair.0 >= 2 - 1.0 / 48
          }
        }
      }
    }
    return true
  }

  var audioInputSummary: String {
    audioInputURL?.path ?? "未指定"
  }

  var musicVideoSummary: String {
    guard let audioDurationSeconds else { return "音源の長さを確認中" }
    let available = max(0, audioDurationSeconds - audioStartSeconds)
    if !usesAutomaticMusicCuts {
      return String(
        format: "全長 %.1f秒・手動ポイント %d件（%d構図）",
        locale: Locale(identifier: "ja_JP"),
        available,
        musicCutPoints.count,
        musicCutPoints.count + 1
      )
    }
    return String(
      format: "全長 %.1f秒（ショット構成は開始時の音源解析で決定・完成済み部分から再開可能）",
      locale: Locale(identifier: "ja_JP"),
      available
    )
  }

  func chooseAudioInput() {
    guard !isRunning, isImageSequence else { return }
    stopAudioPreview()
    let panel = NSOpenPanel()
    panel.title = "H3で使う音源を選択"
    panel.canChooseFiles = true
    panel.canChooseDirectories = false
    panel.allowsMultipleSelection = false
    panel.allowedContentTypes = [.audio, .movie]
    guard panel.runModal() == .OK, let url = panel.url else { return }
    audioInputURL = url.standardizedFileURL
    audioDurationSeconds = nil
    audioStartSeconds = 0
    musicCutPoints = []
    selectedMusicCutPointID = nil
    musicPreviewSeconds = nil
    if !isDurationFixed {
      duration = min(duration, Self.maximumAudioConditioningDuration)
    }
    status = "音源の長さを確認中"
    let selectedURL = url.standardizedFileURL
    Task { [weak self] in
      do {
        let seconds = try await AVURLAsset(url: selectedURL).load(.duration).seconds
        guard let self, self.audioInputURL == selectedURL else { return }
        guard seconds.isFinite, seconds > 0 else {
          self.status = "音源の長さを取得できません"
          return
        }
        self.audioDurationSeconds = seconds
        self.status = self.audioConditioningMode == .lipSync
          ? "外部音源をリップシンク条件に使用します"
          : "外部音源をBGM参照として使用します"
      } catch {
        guard let self, self.audioInputURL == selectedURL else { return }
        self.status = "音源を読み込めません"
        self.appendLog("音源: \(error.localizedDescription)\n")
      }
    }
  }

  func clearAudioInput() {
    guard !isRunning else { return }
    stopAudioPreview()
    audioInputURL = nil
    audioDurationSeconds = nil
    audioStartSeconds = 0
    lyricsText = ""
    musicVideoMode = false
    audioConditioningMode = .backgroundMusic
    musicCutPoints = []
    selectedMusicCutPointID = nil
    musicPreviewSeconds = nil
  }

  var availableMusicDuration: Double {
    max(0, (audioDurationSeconds ?? 0) - audioStartSeconds)
  }

  var selectedMusicCutPointSeconds: Double? {
    guard let selectedMusicCutPointID else { return nil }
    return musicCutPoints.first(where: { $0.id == selectedMusicCutPointID })?.seconds
  }

  @discardableResult
  func addMusicCutPoint(at requestedSeconds: Double) -> Bool {
    guard !isRunning, availableMusicDuration >= 4 else { return false }
    let proposed = quantizedMusicTime(requestedSeconds)
    let all = musicCutPoints.map(\.seconds).sorted()
    guard all.allSatisfy({ abs($0 - proposed) >= 2 }) else {
      status = "構図変更ポイントは前後2秒以上あけてください"
      return false
    }
    let previous = all.last(where: { $0 < proposed }) ?? 0
    let next = all.first(where: { $0 > proposed }) ?? availableMusicDuration
    guard proposed - previous >= 2, next - proposed >= 2 else {
      status = "構図変更ポイントは前後2秒以上あけてください"
      return false
    }
    let point = MiniMaxH3MusicCutPoint(seconds: proposed)
    musicCutPoints.append(point)
    musicCutPoints.sort { $0.seconds < $1.seconds }
    selectedMusicCutPointID = point.id
    return true
  }

  func setMusicPreview(at requestedSeconds: Double, automaticallyStop: Bool) {
    guard !isRunning else { return }
    musicPreviewSeconds = quantizedMusicTime(requestedSeconds)
    previewMusic(
      at: musicPreviewSeconds ?? requestedSeconds,
      automaticallyStop: automaticallyStop
    )
  }

  func commitMusicPreviewPoint() {
    guard let musicPreviewSeconds else { return }
    if addMusicCutPoint(at: musicPreviewSeconds) {
      stopAudioPreview()
    }
  }

  var canCommitMusicPreviewPoint: Bool {
    guard let musicPreviewSeconds,
      musicPreviewSeconds >= 2,
      availableMusicDuration - musicPreviewSeconds >= 2
    else { return false }
    return musicCutPoints.allSatisfy {
      abs($0.seconds - musicPreviewSeconds) >= 2
    }
  }

  var musicPreviewIsCommitted: Bool {
    guard let musicPreviewSeconds else { return false }
    return musicCutPoints.contains {
      abs($0.seconds - musicPreviewSeconds) < 1.0 / 48
    }
  }

  func finishMusicPreview() {
    guard let musicPreviewSeconds else { return }
    previewMusic(at: musicPreviewSeconds, automaticallyStop: true)
  }

  func moveMusicCutPoint(id: UUID, to requestedSeconds: Double) {
    guard !isRunning,
      let currentIndex = musicCutPoints.firstIndex(where: { $0.id == id })
    else { return }
    let others = musicCutPoints.enumerated()
      .filter { $0.offset != currentIndex }
      .map(\.element.seconds)
      .sorted()
    let requested = quantizedMusicTime(requestedSeconds)
    let previous = others.last(where: { $0 <= requested }) ?? 0
    let next = others.first(where: { $0 >= requested }) ?? availableMusicDuration
    let lower = previous + 2
    let upper = next - 2
    guard lower <= upper else { return }
    let movedSeconds = quantizedMusicTime(
      min(upper, max(lower, requested))
    )
    musicCutPoints[currentIndex].seconds = movedSeconds
    musicCutPoints.sort { $0.seconds < $1.seconds }
    selectedMusicCutPointID = id
    musicPreviewSeconds = movedSeconds
  }

  func selectMusicCutPoint(_ id: UUID) {
    selectedMusicCutPointID = id
    musicPreviewSeconds = musicCutPoints.first(where: { $0.id == id })?.seconds
  }

  func removeSelectedMusicCutPoint() {
    guard !isRunning, let selectedMusicCutPointID else { return }
    musicCutPoints.removeAll { $0.id == selectedMusicCutPointID }
    self.selectedMusicCutPointID = musicCutPoints.first?.id
    musicPreviewSeconds = musicCutPoints.first?.seconds
  }

  func clearMusicCutPoints() {
    guard !isRunning else { return }
    musicCutPoints = []
    selectedMusicCutPointID = nil
    musicPreviewSeconds = nil
    stopAudioPreview()
  }

  func audioStartDidChange() {
    stopAudioPreview()
    let duration = availableMusicDuration
    musicPreviewSeconds = nil
    musicCutPoints.removeAll { $0.seconds < 2 || $0.seconds > duration - 2 }
    if let selectedMusicCutPointID,
      !musicCutPoints.contains(where: { $0.id == selectedMusicCutPointID })
    {
      self.selectedMusicCutPointID = musicCutPoints.first?.id
    }
  }

  private func quantizedMusicTime(_ seconds: Double) -> Double {
    (min(availableMusicDuration, max(0, seconds)) * 24).rounded() / 24
  }

  func previewMusic(at relativeSeconds: Double, automaticallyStop: Bool) {
    guard !isRunning, let audioInputURL else { return }
    let relative = quantizedMusicTime(relativeSeconds)
    let absolute = audioStartSeconds + relative
    audioPreviewStopTask?.cancel()
    audioPreviewStopTask = nil
    if audioPreviewURL != audioInputURL || audioPreviewPlayer == nil {
      audioPreviewPlayer?.pause()
      audioPreviewPlayer = AVPlayer(url: audioInputURL)
      audioPreviewURL = audioInputURL
      lastAudioPreviewSeconds = nil
    }
    guard let audioPreviewPlayer else { return }
    if lastAudioPreviewSeconds.map({ abs($0 - absolute) >= 0.08 }) ?? true {
      audioPreviewPlayer.seek(
        to: CMTime(seconds: absolute, preferredTimescale: 600),
        toleranceBefore: CMTime(seconds: 0.02, preferredTimescale: 600),
        toleranceAfter: CMTime(seconds: 0.02, preferredTimescale: 600)
      )
      lastAudioPreviewSeconds = absolute
    }
    audioPreviewPlayer.play()
    if automaticallyStop {
      audioPreviewStopTask = Task { @MainActor [weak self] in
        try? await Task.sleep(for: .seconds(1.5))
        guard !Task.isCancelled else { return }
        self?.stopAudioPreview()
      }
    }
  }

  func stopAudioPreview() {
    audioPreviewStopTask?.cancel()
    audioPreviewStopTask = nil
    audioPreviewPlayer?.pause()
    lastAudioPreviewSeconds = nil
  }

  func prepare(upscalerInput: URL?) {
    guard upscalerInput != currentUpscalerInput else { return }
    currentUpscalerInput = upscalerInput
    guard usesUpscalerInput, !supportsPromptOnly else { return }
    setInputURLs(
      upscalerInput.map { [$0.standardizedFileURL] } ?? [],
      upscalerInput: true
    )
  }

  func useUpscalerInput(_ input: URL?) {
    currentUpscalerInput = input
    guard !supportsPromptOnly else { return }
    setInputURLs(
      input.map { [$0.standardizedFileURL] } ?? [],
      upscalerInput: true
    )
  }

  func chooseInput() {
    guard !supportsPromptOnly else {
      status = "プロンプトのみでは参照入力を使用しません"
      return
    }
    let panel = NSOpenPanel()
    panel.title = "MiniMax H3の入力動画または画像を選択"
    panel.canChooseFiles = true
    panel.canChooseDirectories = false
    panel.allowsMultipleSelection = true
    panel.allowedContentTypes = [.movie, .image]
    guard panel.runModal() == .OK else { return }
    var urls = panel.urls.map(\.standardizedFileURL)
    let allImages = !urls.isEmpty && urls.allSatisfy(Self.isImage)
    if urls.count > 1, !allImages {
      status = "複数選択できるのは画像だけです"
      return
    }
    if allImages, urls.count > Self.maximumIdentityImages {
      status = "Subject参照画像は最大\(Self.maximumIdentityImages)枚です"
      return
    }
    if allImages {
      urls.sort {
        $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent)
          == .orderedAscending
      }
    }
    setInputURLs(urls, upscalerInput: false)
  }

  private func setInputURLs(_ urls: [URL], upscalerInput: Bool) {
    resetFaceReferences()
    inputURLs = urls
    usesUpscalerInput = upscalerInput
    guard let first = urls.first else { return }
    let suffix = urls.count > 1 ? "-\(urls.count)-images" : ""
    let proposed = first.deletingPathExtension().path
      + suffix + "-minimax-h3.mp4"
    if outputPath.isEmpty || outputPath == automaticOutputPath {
      outputPath = proposed
      automaticOutputPath = proposed
    }
    if imageReferenceScope == .faceOnly,
      !urls.isEmpty,
      urls.allSatisfy(Self.isImage)
    {
      detectFaces()
    }
  }

  private var validInputSelection: Bool {
    if supportsPromptOnly { return inputURLs.isEmpty }
    guard !inputURLs.isEmpty else { return false }
    if isImageSequence, imageReferenceScope == .faceOnly {
      let count = selectedFaceReferences.count
      return !isDetectingFaces && count > 0
        && count <= Self.maximumIdentityImages
    }
    if inputURLs.count == 1 {
      return Self.isImage(inputURLs[0]) || Self.isMovie(inputURLs[0])
    }
    return inputURLs.count <= Self.maximumIdentityImages
      && inputURLs.allSatisfy(Self.isImage)
  }

  var selectedFaceReferenceCount: Int {
    selectedFaceReferences.count
  }

  private var selectedFaceReferences: [MiniMaxH3FaceReference] {
    faceReferences.filter(\.isSelected).sorted {
      if $0.sourceIndex != $1.sourceIndex {
        return $0.sourceIndex < $1.sourceIndex
      }
      return $0.faceIndex < $1.faceIndex
    }
  }

  func selectImageReferenceScope(_ scope: MiniMaxH3ImageReferenceScope) {
    guard !isRunning, imageReferenceScope != scope else { return }
    imageReferenceScope = scope
    if scope == .faceOnly, isImageSequence {
      detectFaces()
    } else if scope == .wholeImage {
      resetFaceReferences()
    }
  }

  func detectFaces() {
    guard !isRunning, isImageSequence else { return }
    resetFaceReferences()
    let sourceURLs = inputURLs
    let directory = FileManager.default.temporaryDirectory
      .appendingPathComponent(
        "mioh-h3-face-references-\(UUID().uuidString)",
        isDirectory: true
      )
    faceReferenceDirectory = directory
    isDetectingFaces = true
    status = "参照画像から顔を検出中"
    faceDetectionTask = Task { [weak self] in
      do {
        let faces = try await MiniMaxH3FaceReferenceProcessor.detectFaces(
          in: sourceURLs,
          destinationDirectory: directory
        )
        guard !Task.isCancelled, let self else { return }
        self.faceReferences = faces
        self.isDetectingFaces = false
        self.faceDetectionTask = nil
        if faces.isEmpty {
          self.status = "顔を検出できませんでした。画像全体を使用してください"
        } else if faces.count > Self.maximumIdentityImages {
          self.status = "\(faces.count)件検出。使用する顔を最大\(Self.maximumIdentityImages)件選択してください"
        } else {
          self.status = "顔を\(faces.count)件検出しました"
        }
      } catch is CancellationError {
        guard let self else { return }
        self.isDetectingFaces = false
        self.faceDetectionTask = nil
      } catch {
        guard let self else { return }
        self.isDetectingFaces = false
        self.faceDetectionTask = nil
        self.status = "顔検出に失敗しました"
        self.appendLog("顔検出: \(error.localizedDescription)\n")
      }
    }
  }

  func setFaceReferenceSelected(_ id: String, selected: Bool) {
    guard !isRunning,
      let index = faceReferences.firstIndex(where: { $0.id == id })
    else { return }
    if selected, selectedFaceReferences.count >= Self.maximumIdentityImages {
      status = "使用できる顔参照は最大\(Self.maximumIdentityImages)件です"
      return
    }
    faceReferences[index].isSelected = selected
  }

  func setFaceReferenceSubject(_ id: String, subjectIndex: Int) {
    guard !isRunning,
      (1...Self.maximumIdentityImages).contains(subjectIndex),
      let index = faceReferences.firstIndex(where: { $0.id == id })
    else { return }
    faceReferences[index].subjectIndex = subjectIndex
  }

  func groupSelectedFacesAsOneSubject() {
    guard !isRunning else { return }
    for index in faceReferences.indices where faceReferences[index].isSelected {
      faceReferences[index].subjectIndex = 1
    }
    status = "選択した顔を<Subject 1>へまとめました"
  }

  private func resetFaceReferences() {
    faceDetectionTask?.cancel()
    faceDetectionTask = nil
    isDetectingFaces = false
    faceReferences = []
    if let faceReferenceDirectory {
      try? FileManager.default.removeItem(at: faceReferenceDirectory)
    }
    faceReferenceDirectory = nil
  }

  private func faceReferencePrompt(_ originalPrompt: String) -> String {
    MiniMaxH3FaceReferenceProcessor.faceOnlyPrompt(
      originalPrompt,
      references: selectedFaceReferences
    )
  }

  private func audioConditionedPrompt(_ originalPrompt: String) -> String {
    guard audioInputURL != nil else { return originalPrompt }
    switch audioConditioningMode {
    case .lipSync:
      return originalPrompt
    case .backgroundMusic:
      let directive = """

        Audio reference directive: Use the supplied audio only as non-diegetic background music and music-video structure reference. Follow its tempo, energy, section changes, instrumentation, vocal mood, and emotional dynamics, but do not generate lip-sync, singing mouth shapes, dialogue performance, or visible speech from this audio. On-screen people must not sing to the vocals; their lips stay closed or move only naturally with breathing, expression, or non-vocal acting.
        """
      return originalPrompt + directive
    }
  }

  func chooseLyricsFile() {
    guard !isRunning else { return }
    let panel = NSOpenPanel()
    panel.title = "歌詞テキストを選択"
    panel.canChooseFiles = true
    panel.canChooseDirectories = false
    panel.allowsMultipleSelection = false
    panel.allowedContentTypes = [.plainText, .utf8PlainText, .text]
    guard panel.runModal() == .OK, let url = panel.url else { return }
    do {
      lyricsText = try String(contentsOf: url, encoding: .utf8)
      if lyricsSearchQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        .isEmpty
      {
        lyricsSearchQuery = url.deletingPathExtension().lastPathComponent
      }
      status = "歌詞を読み込みました"
    } catch {
      status = "歌詞を読み込めません"
      appendLog("歌詞: \(error.localizedDescription)\n")
    }
  }

  func openLyricsSearch() {
    let query = lyricsSearchQuery.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !query.isEmpty else {
      status = "曲名やアーティスト名を入力してください"
      return
    }
    var components = URLComponents(string: "https://www.google.com/search")
    components?.queryItems = [
      URLQueryItem(name: "q", value: "\(query) lyrics 歌詞")
    ]
    if let url = components?.url {
      NSWorkspace.shared.open(url)
      status = "歌詞検索をブラウザで開きました"
    }
  }

  private func lyricsConditionedPrompt(_ originalPrompt: String) -> String {
    let lyrics = lyricsText.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !lyrics.isEmpty else { return originalPrompt }
    let directive: String
    switch audioConditioningMode {
    case .backgroundMusic:
      directive = "Treat these lyrics as song meaning, emotional timing, imagery, and section guidance only. Do not generate lip-sync, singing mouth shapes, visible speech, karaoke subtitles, lyric cards, or on-screen text unless the interval body explicitly asks for visible text."
    case .lipSync:
      directive = "Use timed lyric lines as lip-sync and expression timing guidance when an interval explicitly calls for singing or visible vocal performance. Keep the exact language of lyric snippets."
    }
    return originalPrompt + """

      LYRICS / SONG MEANING:
      \(directive)
      \(lyrics)
      """
  }

  private static func isImage(_ url: URL) -> Bool {
    UTType(filenameExtension: url.pathExtension)?.conforms(to: .image) == true
  }

  private static func isMovie(_ url: URL) -> Bool {
    UTType(filenameExtension: url.pathExtension)?.conforms(to: .movie) == true
  }

  private static func resolvePipelineManifestPath(_ path: String) -> String {
    guard !path.isEmpty else { return path }
    let selected = URL(fileURLWithPath: path).standardizedFileURL
    if isPipelineManifest(selected) { return selected.path }
    let sibling = selected.deletingLastPathComponent()
      .appendingPathComponent("manifest.json")
      .standardizedFileURL
    return isPipelineManifest(sibling) ? sibling.path : selected.path
  }

  private static func isPipelineManifest(_ url: URL) -> Bool {
    guard let data = try? Data(contentsOf: url),
      let object = try? JSONSerialization.jsonObject(with: data),
      let dictionary = object as? [String: Any]
    else { return false }
    return dictionary["schemaVersion"] != nil
      && dictionary["modelIdentifier"] != nil
      && dictionary["stages"] != nil
      && dictionary["sigmas"] != nil
  }

  private static func isPromptOnlyManifest(_ path: String) -> Bool {
    guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
      let object = try? JSONSerialization.jsonObject(with: data),
      let dictionary = object as? [String: Any]
    else { return false }
    if let mode = dictionary["conditioningMode"] as? String {
      return mode == "fl2va"
    }
    let identifier = dictionary["modelIdentifier"] as? String ?? ""
    return identifier.localizedCaseInsensitiveContains("fl2va")
  }

  private func refreshConditioningMode() {
    let promptOnly = Self.isPromptOnlyManifest(manifestPath)
    guard supportsPromptOnly != promptOnly else { return }
    supportsPromptOnly = promptOnly
    if promptOnly {
      resetFaceReferences()
      inputURLs = []
      usesUpscalerInput = false
      ensurePromptOnlyOutputPath()
    }
  }

  func selectGenerationMode(promptOnly: Bool) {
    guard !isRunning else { return }
    let current = URL(fileURLWithPath: effectiveManifestPath)
      .standardizedFileURL
    let filename = promptOnly ? "manifest-fl2va.json" : "manifest.json"
    let candidate = current.deletingLastPathComponent()
      .appendingPathComponent(filename)
      .standardizedFileURL
    guard Self.isPipelineManifest(candidate),
      Self.isPromptOnlyManifest(candidate.path) == promptOnly
    else {
      status = promptOnly
        ? "FL2VA prompt-onlyモデルが未変換です"
        : "Ref2VA参照モデルが見つかりません"
      return
    }
    manifestPath = candidate.path
    if promptOnly {
      resetFaceReferences()
      inputURLs = []
      usesUpscalerInput = false
      ensurePromptOnlyOutputPath()
      status = "プロンプトのみ（FL2VA）"
    } else {
      usesUpscalerInput = true
      setInputURLs(
        currentUpscalerInput.map { [$0.standardizedFileURL] } ?? [],
        upscalerInput: true
      )
      status = "参照画像／動画（Ref2VA）"
    }
  }

  private func ensurePromptOnlyOutputPath() {
    guard outputPath.isEmpty || outputPath == automaticOutputPath else { return }
    let desktop = FileManager.default.urls(
      for: .desktopDirectory,
      in: .userDomainMask
    ).first!
    let proposed = desktop.appendingPathComponent(
      "10eros-max-h3-prompt-(seed).mp4"
    ).path
    outputPath = proposed
    automaticOutputPath = proposed
  }

  func chooseManifest() {
    let panel = NSOpenPanel()
    panel.title = "MiniMax H3 Swiftマニフェストを選択"
    panel.canChooseFiles = true
    panel.canChooseDirectories = false
    panel.allowsMultipleSelection = false
    panel.allowedContentTypes = [.json]
    if panel.runModal() == .OK, let url = panel.url {
      let selected = url.standardizedFileURL.path
      let resolved = Self.resolvePipelineManifestPath(selected)
      guard Self.isPipelineManifest(URL(fileURLWithPath: resolved)) else {
        status = "MiniMax H3のパイプラインmanifest.jsonを選択してください"
        return
      }
      manifestPath = resolved
      if selected != resolved {
        status = "同じフォルダのパイプラインmanifest.jsonへ補正しました"
      }
    }
  }

  func chooseMusicVideoLastFrameDirectory() {
    guard !isRunning else { return }
    let panel = NSOpenPanel()
    panel.title = "Codexが作成したPart終端画像のフォルダを選択"
    panel.canChooseFiles = false
    panel.canChooseDirectories = true
    panel.canCreateDirectories = true
    panel.allowsMultipleSelection = false
    if panel.runModal() == .OK, let url = panel.url {
      musicVideoLastFrameDirectory = url.standardizedFileURL.path
    }
  }

  func chooseCoreAICacheRoot() {
    let panel = NSOpenPanel()
    panel.title = "Core AIキャッシュ保存先を選択"
    panel.canChooseFiles = false
    panel.canChooseDirectories = true
    panel.canCreateDirectories = true
    panel.allowsMultipleSelection = false
    if !coreAICacheRoot.isEmpty {
      panel.directoryURL = URL(fileURLWithPath: coreAICacheRoot, isDirectory: true)
    }
    guard panel.runModal() == .OK, let url = panel.url else { return }
    coreAICacheRoot = url.standardizedFileURL.path
  }

  func useSystemCoreAICache() {
    coreAICacheRoot = ""
  }

  func clearCoreAICache() {
    guard !isRunning, let directory = effectiveCoreAICacheDirectory else { return }
    do {
      if FileManager.default.fileExists(atPath: directory.path) {
        try FileManager.default.removeItem(at: directory)
      }
      try FileManager.default.createDirectory(
        at: directory,
        withIntermediateDirectories: true
      )
      status = "Core AIキャッシュを削除しました"
      refreshCoreAICacheSummary()
    } catch {
      status = "Core AIキャッシュを削除できませんでした"
      appendLog("Core AI cache: \(error.localizedDescription)\n")
    }
  }

  private var effectiveCoreAICacheDirectory: URL? {
    let value = coreAICacheRoot.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !value.isEmpty else { return nil }
    return URL(fileURLWithPath: value, isDirectory: true)
      .standardizedFileURL
      .appendingPathComponent("mioh-coreai-cache", isDirectory: true)
  }

  private func refreshCoreAICacheSummary() {
    guard let directory = effectiveCoreAICacheDirectory else {
      coreAICacheSummary = "システム標準"
      return
    }
    coreAICacheSummary = "容量を確認中"
    Task { [weak self] in
      let bytes = await Task.detached(priority: .utility) {
        Self.directoryByteCount(directory)
      }.value
      guard let self, self.effectiveCoreAICacheDirectory == directory else { return }
      self.coreAICacheSummary = "\(directory.path) · \(ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file))"
    }
  }

  nonisolated private static func directoryByteCount(_ directory: URL) -> Int64 {
    guard let enumerator = FileManager.default.enumerator(
      at: directory,
      includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey],
      options: [.skipsHiddenFiles]
    ) else { return 0 }
    var total: Int64 = 0
    for case let url as URL in enumerator {
      guard let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
        values.isRegularFile == true
      else { continue }
      total += Int64(values.fileSize ?? 0)
    }
    return total
  }

  func chooseOutput() {
    let panel = NSSavePanel()
    panel.title = "H3生成動画の保存先"
    panel.nameFieldStringValue = URL(fileURLWithPath: outputPath).lastPathComponent
    panel.allowedContentTypes = [.mpeg4Movie]
    if panel.runModal() == .OK, let url = panel.url {
      outputPath = url.standardizedFileURL.path
      automaticOutputPath = ""
    }
  }

  func start() {
    if let fixedDuration = selectedResolutionProfile?.fixedDuration {
      duration = fixedDuration
    } else {
      duration = min(
        maximumShotDuration,
        max(2, (duration * 2).rounded() / 2)
      )
    }
    guard canStart(), let runnerURL else {
      if !supportsRuntime {
        status = "macOS 27が必要です"
      } else if !modelReady {
        status = "変換済みMiniMax H3モデルが未インストールです"
      } else {
        status = "H3設定を確認してください"
      }
      return
    }
    let cachesRoot = FileManager.default.urls(
      for: .cachesDirectory,
      in: .userDomainMask
    ).first!
    let preferredCache = cachesRoot.appendingPathComponent(
      "com.okatti.mioh.upscaler/10eros-max-h3",
      isDirectory: true
    )
    let legacyCache = cachesRoot.appendingPathComponent(
      "com.okatti.lada.coreai/10eros-max-h3",
      isDirectory: true
    )
    // Preserve completed stage caches created by earlier Mioh builds. New
    // installations use the upscaler-owned path.
    let cache = FileManager.default.fileExists(atPath: legacyCache.path)
      && !FileManager.default.fileExists(atPath: preferredCache.path)
      ? legacyCache : preferredCache
    do {
      try FileManager.default.createDirectory(
        at: cache,
        withIntermediateDirectories: true
      )
    } catch {
      status = "キャッシュ作成失敗"
      appendLog("\(error.localizedDescription)\n")
      return
    }
    let task = Process()
    let outputPipe = Pipe()
    let errorPipe = Pipe()
    task.executableURL = runnerURL
    let resolvedManifestPath = effectiveManifestPath
    if manifestPath != resolvedManifestPath {
      manifestPath = resolvedManifestPath
    }
    let runtimeImageURLs = isImageSequence
      && imageReferenceScope == .faceOnly
      ? selectedFaceReferences.map(\.cropURL)
      : inputURLs
    let baseRuntimePrompt = isImageSequence
      && imageReferenceScope == .faceOnly
      ? audioConditionedPrompt(faceReferencePrompt(prompt))
      : audioConditionedPrompt(prompt)
    let runtimePrompt = lyricsConditionedPrompt(baseRuntimePrompt)
    var arguments = [
      musicVideoMode ? "music-video" : "run",
      "--manifest", resolvedManifestPath,
      "--output", outputPath,
      "--prompt", runtimePrompt,
      "--cache", cache.path,
      "--backend", backend,
      "--width", String(width),
      "--height", String(height),
      "--output-width", String(outputWidth),
      "--output-height", String(outputHeight),
      "--duration", String(duration),
      "--seed", seed,
    ]
    if musicVideoMode {
      arguments += [
        "--music-video-continuation", musicVideoContinuationMode.rawValue,
      ]
      if musicVideoContinuationMode == .firstAndProvidedLast {
        arguments += [
          "--music-video-last-frame-directory", musicVideoLastFrameDirectory,
        ]
      }
    }
    if let audioInputURL {
      arguments += [
        "--audio-input", audioInputURL.path,
        "--audio-start", String(audioStartSeconds),
        "--audio-conditioning-mode", audioConditioningMode.rawValue,
      ]
      if musicVideoMode, !usesAutomaticMusicCuts {
        do {
          let data = try JSONEncoder().encode(
            musicCutPoints.map(\.seconds).sorted()
          )
          arguments += [
            "--music-video-cuts-json",
            String(decoding: data, as: UTF8.self),
          ]
        } catch {
          status = "構図変更ポイントの準備に失敗しました"
          appendLog("\(error.localizedDescription)\n")
          return
        }
      }
    }
    if isImageSequence {
      do {
        let data = try JSONEncoder().encode(runtimeImageURLs.map(\.path))
        arguments += [
          "--input-images-json",
          String(decoding: data, as: UTF8.self),
        ]
      } catch {
        status = "画像入力の準備に失敗しました"
        appendLog("\(error.localizedDescription)\n")
        return
      }
    } else if let video = inputURLs.first {
      arguments += ["--input", video.path]
    }
    task.arguments = arguments
    var environment = ProcessInfo.processInfo.environment
    if let coreAICacheDirectory = effectiveCoreAICacheDirectory {
      do {
        try FileManager.default.createDirectory(
          at: coreAICacheDirectory,
          withIntermediateDirectories: true
        )
        environment["MIOH_H3_COREAI_CACHE_ROOT"] = coreAICacheDirectory.path
      } catch {
        status = "Core AIキャッシュ作成失敗"
        appendLog("\(error.localizedDescription)\n")
        return
      }
    } else {
      environment.removeValue(forKey: "MIOH_H3_COREAI_CACHE_ROOT")
    }
    task.environment = environment
    task.standardOutput = outputPipe
    task.standardError = errorPipe
    outputPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
      let data = handle.availableData
      guard !data.isEmpty else { return }
      Task { @MainActor in self?.consumeStandardOutput(data) }
    }
    errorPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
      let data = handle.availableData
      guard !data.isEmpty else { return }
      Task { @MainActor in
        self?.consumeStandardError(data)
      }
    }
    task.terminationHandler = { [weak self] completed in
      Task { @MainActor in
        outputPipe.fileHandleForReading.readabilityHandler = nil
        errorPipe.fileHandleForReading.readabilityHandler = nil
        guard let self else { return }
        self.process = nil
        self.isRunning = false
        if completed.terminationStatus == 0 {
          self.progress = 1
          self.status = "MiniMax H3生成完了"
        } else if self.status == "停止中" {
          self.status = "停止"
        } else {
          self.status = "MiniMax H3生成失敗"
        }
      }
    }
    do {
      progress = 0
      status = "MiniMax H3準備中"
      log = "Swift / \(backend == "coreai" ? "Core AI" : "Core ML")\n"
      appendLog(
        effectiveCoreAICacheDirectory.map { "Core AI cache: \($0.path)\n" }
          ?? "Core AI cache: system default\n"
      )
      musicAnalysisSummary = musicVideoMode ? "音源解析を開始します" : ""
      if musicVideoMode {
        appendLog("[musicAnalysis] queued: 音源解析を開始します\n")
      }
      standardOutputBuffer.removeAll(keepingCapacity: true)
      standardErrorBuffer.removeAll(keepingCapacity: true)
      try task.run()
      process = task
      isRunning = true
    } catch {
      status = "MiniMax H3起動失敗"
      appendLog("\(error.localizedDescription)\n")
    }
  }

  func stop() {
    guard let process, process.isRunning else { return }
    status = "停止中"
    process.interrupt()
    DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + 2) {
      if process.isRunning { process.terminate() }
    }
  }

  func revealOutput() {
    guard !outputPath.isEmpty else { return }
    NSWorkspace.shared.activateFileViewerSelecting([
      URL(fileURLWithPath: outputPath)
    ])
  }

  private func consumeStandardOutput(_ data: Data) {
    standardOutputBuffer.append(data)
    while let newline = standardOutputBuffer.firstIndex(of: 0x0A) {
      let lineData = standardOutputBuffer.prefix(upTo: newline)
      standardOutputBuffer.removeSubrange(...newline)
      guard !lineData.isEmpty else { continue }
      if let event = try? JSONDecoder().decode(
        MiniMaxH3UIProgressEvent.self,
        from: Data(lineData)
      ) {
        progress = event.progress
        status = event.message
        if event.stage == "musicAnalysis" {
          musicAnalysisSummary = event.message
        }
        appendLog("[\(event.stage)] \(event.state): \(event.message)\n")
      } else {
        let line = String(decoding: lineData, as: UTF8.self)
        // Core AI/MPSGraph can emit compiler diagnostics on stdout as well as
        // stderr. Keep the user-facing log limited to pipeline progress and
        // actionable runner errors.
        guard !isInternalCoreAIWarning(line) else { continue }
        appendLog(line + "\n")
      }
    }
  }

  private func consumeStandardError(_ data: Data) {
    standardErrorBuffer.append(data)
    while let newline = standardErrorBuffer.firstIndex(of: 0x0A) {
      let lineData = standardErrorBuffer.prefix(upTo: newline)
      standardErrorBuffer.removeSubrange(...newline)
      guard !lineData.isEmpty else { continue }
      let line = String(decoding: lineData, as: UTF8.self)
      if isInternalCoreAIWarning(line) { continue }
      appendLog(line + "\n")
    }
  }

  private func isInternalCoreAIWarning(_ line: String) -> Bool {
    let normalized = line.lowercased()
    let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
    let fragments = [
      "#aicode.",
      "aicode.serialization",
      "ane_validation_message",
      "anecompiler",
      "aneccompile(",
      "mlir mps to anec",
      "failed: ane i/o op",
      "incompatible element type for ane",
      "ane compilation failed",
      "full compile with ane as preferred device failed",
      "no ane hash for architecture",
      "gpu-only model or wrong target",
    ]
    if fragments.contains(where: { normalized.contains($0) }) {
      return true
    }

    // On macOS 27, the serialized ANE diagnostic is sometimes split across
    // writes. Once its aicode-bearing body is removed above, Core AI leaves
    // these otherwise meaningless continuation lines behind:
    //   2026-... mioh-minimax-h3-native[pid:thread]
    //   Error:
    //   )}}}
    // Real runner failures use `mioh-minimax-h3-native: <message>` on one
    // line, so these signatures can be discarded without hiding them.
    if normalized.contains("mioh-minimax-h3-native[") {
      return true
    }
    if normalized == "error:" || normalized == "warning:" {
      return true
    }
    let diagnosticPunctuation = CharacterSet(charactersIn: "(){}[]<>,:#=)")
    if !trimmed.isEmpty,
      trimmed.unicodeScalars.allSatisfy({ diagnosticPunctuation.contains($0) })
    {
      return true
    }
    return false
  }

  private func appendLog(_ text: String) {
    log += text
    guard log.count > Self.maximumVisibleLogCharacters else { return }
    log = "…以前のログを省略…\n" + log.suffix(Self.retainedLogCharacters)
  }

  func generateAIPrompt() {
    guard !isGeneratingAIPrompt else { return }
    let request = aiPromptRequest.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !request.isEmpty else {
      status = "AIへの指示を書いてください"
      return
    }
    isGeneratingAIPrompt = true
    status = "AIプロンプト生成中"
    Task {
      do {
        let generated = try await requestAIPrompt(request)
        aiGeneratedPrompt = generated
        status = "AIプロンプトを生成しました"
      } catch {
        status = "AIプロンプト生成に失敗しました"
        appendLog("AIプロンプト生成: \(error.localizedDescription)\n")
      }
      isGeneratingAIPrompt = false
    }
  }

  func applyGeneratedAIPrompt() {
    let generated = aiGeneratedPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !generated.isEmpty else {
      status = "AI生成プロンプトが空です"
      return
    }
    prompt = generated
    status = "AI生成プロンプトをMiniMaxプロンプト欄へ反映しました"
  }

  func applyAIPromptProviderDefaults() {
    aiPromptAPIURL = aiPromptProvider.defaultBaseURL
  }

  private func aiSubjectReferenceSummary() -> String {
    guard imageReferenceScope == .faceOnly, !selectedFaceReferences.isEmpty
    else {
      return """
        SUBJECT REFERENCES:
        - none selected for AI prompt generation.
        """
    }
    var pictureLabelsBySubject: [Int: [String]] = [:]
    for (index, face) in selectedFaceReferences.enumerated() {
      pictureLabelsBySubject[face.subjectIndex, default: []]
        .append("<Picture \(index + 1)>")
    }
    let lines = pictureLabelsBySubject.keys.sorted().map { subject in
      let pictures = pictureLabelsBySubject[subject, default: []]
        .joined(separator: ", ")
      return "  - <Subject \(subject)>: facial identity comes from \(pictures); use this exact subject label in subject_definitions and interval bodies."
    }.joined(separator: "\n")
    return """
      SUBJECT REFERENCES:
      \(lines)
      - Use labels such as <Subject 1> as silent control metadata, not visible text.
      - Do not replace <Subject 1> with generic names like protagonist, singer, man, woman, actor, or character when referring to the referenced person.
      """
  }

  private func aiMusicAnalysisSummary(
    targetDuration: Double,
    shotLimitSeconds: Double
  ) async -> String {
    guard musicVideoMode else {
      return "AUDIO ANALYSIS: not used because long music-video mode is off."
    }
    guard let audioInputURL else {
      return "AUDIO ANALYSIS: unavailable because no audio file is selected."
    }
    guard targetDuration >= 2 else {
      return "AUDIO ANALYSIS: unavailable because selected audio duration is too short."
    }
    do {
      let samples = try await readAIPromptAnalysisAudio(
        url: audioInputURL,
        startSeconds: audioStartSeconds,
        durationSeconds: targetDuration,
        sampleRate: 8_000
      )
      let intervals = analyzedAIPromptIntervals(
        samples: samples,
        sampleRate: 8_000,
        totalDuration: targetDuration,
        shotLimitSeconds: shotLimitSeconds
      )
      let intervalLines = intervals.map { interval in
        String(
          format:
            "  - [%.3f-%.3f %@] energy %.2f, boundary %.2f",
          locale: Locale(identifier: "en_US_POSIX"),
          interval.start,
          interval.end,
          interval.transition,
          interval.energy,
          interval.change
        )
      }
      .joined(separator: "\n")
      let sectionLines = summarizedAIPromptMusicSections(intervals)
      return """
        AUDIO ANALYSIS:
        - analyzed_audio_file: \(audioInputURL.lastPathComponent)
        - analyzed_start_seconds: \(String(format: "%.3f", audioStartSeconds))
        - analyzed_total_seconds: \(String(format: "%.3f", targetDuration))
        - analysis_mode: lightweight UI analysis of RMS energy, energy deltas, and local novelty\(usesAutomaticMusicCuts ? "" : " with user composition points preferred")
        - Use the logical sections for story pacing, emotional arc, scene choice, and lyric emphasis.
        - Use the suggested generation intervals exactly as bracket markers in the final prompt unless the user's instruction explicitly overrides them.
        Logical music sections:
        \(sectionLines)
        Suggested generation intervals:
        \(intervalLines)
        """
    } catch {
      return """
        AUDIO ANALYSIS:
        - unavailable: \(error.localizedDescription)
        - fallback_instruction: Still write the full target_total_seconds timeline. Use 8-10 second intervals, infer verse/pre-chorus/chorus/bridge/outro from lyrics, and include [start-end cut] or [start-end continue] on every marker.
        """
    }
  }

  private struct AIPromptMusicInterval {
    let start: Double
    let end: Double
    let transition: String
    let energy: Double
    let change: Double
  }

  private func readAIPromptAnalysisAudio(
    url: URL,
    startSeconds: Double,
    durationSeconds: Double,
    sampleRate: Int
  ) async throws -> [Float] {
    let asset = AVURLAsset(url: url)
    guard let track = try await asset.loadTracks(withMediaType: .audio).first else {
      throw NSError(
        domain: "AIPromptAudioAnalysis",
        code: -1,
        userInfo: [NSLocalizedDescriptionKey: "audio track not found"]
      )
    }
    return try await Task.detached(priority: .userInitiated) {
      let reader = try AVAssetReader(asset: asset)
      reader.timeRange = CMTimeRange(
        start: CMTime(seconds: startSeconds, preferredTimescale: 600),
        duration: CMTime(seconds: durationSeconds, preferredTimescale: 600)
      )
      let output = AVAssetReaderTrackOutput(
        track: track,
        outputSettings: [
          AVFormatIDKey: kAudioFormatLinearPCM,
          AVSampleRateKey: sampleRate,
          AVNumberOfChannelsKey: 2,
          AVLinearPCMBitDepthKey: 32,
          AVLinearPCMIsFloatKey: true,
          AVLinearPCMIsBigEndianKey: false,
          AVLinearPCMIsNonInterleaved: false,
        ]
      )
      reader.add(output)
      guard reader.startReading() else {
        throw reader.error
          ?? NSError(
            domain: "AIPromptAudioAnalysis",
            code: -2,
            userInfo: [NSLocalizedDescriptionKey: "audio reader did not start"]
          )
      }
      let maximumFrames = max(
        1,
        Int((durationSeconds * Double(sampleRate)).rounded())
      )
      var mono: [Float] = []
      mono.reserveCapacity(maximumFrames)
      while mono.count < maximumFrames,
        let sampleBuffer = output.copyNextSampleBuffer()
      {
        guard let block = CMSampleBufferGetDataBuffer(sampleBuffer) else {
          continue
        }
        let byteCount = CMBlockBufferGetDataLength(block)
        guard byteCount > 0,
          byteCount % MemoryLayout<Float>.stride == 0
        else { continue }
        var data = Data(count: byteCount)
        let status = data.withUnsafeMutableBytes { raw in
          CMBlockBufferCopyDataBytes(
            block,
            atOffset: 0,
            dataLength: byteCount,
            destination: raw.baseAddress!
          )
        }
        guard status == noErr else { continue }
        data.withUnsafeBytes { raw in
          let values = raw.bindMemory(to: Float.self)
          var index = 0
          while index + 1 < values.count, mono.count < maximumFrames {
            mono.append((values[index] + values[index + 1]) * 0.5)
            index += 2
          }
        }
      }
      if mono.isEmpty {
        throw NSError(
          domain: "AIPromptAudioAnalysis",
          code: -3,
          userInfo: [NSLocalizedDescriptionKey: "no audio samples decoded"]
        )
      }
      return mono
    }.value
  }

  private func analyzedAIPromptIntervals(
    samples: [Float],
    sampleRate: Int,
    totalDuration: Double,
    shotLimitSeconds: Double
  ) -> [AIPromptMusicInterval] {
    let hopSeconds = 0.25
    let hopSamples = max(1, Int((hopSeconds * Double(sampleRate)).rounded()))
    let frameCount = max(1, samples.count / hopSamples)
    var energies: [Double] = []
    energies.reserveCapacity(frameCount)
    for frameIndex in 0..<frameCount {
      let start = frameIndex * hopSamples
      let end = min(samples.count, start + hopSamples)
      guard start < end else { continue }
      var sum = 0.0
      for sample in samples[start..<end] {
        let value = Double(sample)
        sum += value * value
      }
      energies.append(sqrt(sum / Double(end - start)))
    }
    guard !energies.isEmpty else {
      return fallbackAIPromptIntervals(
        totalDuration: totalDuration,
        shotLimitSeconds: shotLimitSeconds
      )
    }
    let maxEnergy = max(energies.max() ?? 0.0001, 0.0001)
    let normalized = energies.map { min(1, $0 / maxEnergy) }
    var novelty = [Double](repeating: 0, count: normalized.count)
    for index in normalized.indices.dropFirst() {
      novelty[index] = abs(normalized[index] - normalized[index - 1])
        + max(0, normalized[index] - normalized[index - 1]) * 0.5
    }

    let manualCuts = usesAutomaticMusicCuts
      ? []
      : musicCutPoints.map(\.seconds)
    var boundaries: [Double]
    if manualCuts.isEmpty {
      boundaries = [0]
      var cursor = 0.0
      while totalDuration - cursor > shotLimitSeconds + 0.5 {
        let ideal = cursor + min(shotLimitSeconds, max(6.0, shotLimitSeconds * 0.9))
        let snapped = strongestAIPromptBoundary(
          near: ideal,
          novelty: novelty,
          hopSeconds: hopSeconds,
          minimum: cursor + 2,
          maximum: min(totalDuration - 2, cursor + shotLimitSeconds)
        )
        if snapped <= cursor + 1.0 { break }
        boundaries.append(snapped)
        cursor = snapped
      }
      if boundaries.last ?? 0 < totalDuration {
        boundaries.append(totalDuration)
      }
    } else {
      boundaries = ([0] + manualCuts + [totalDuration])
        .filter { $0 >= 0 && $0 <= totalDuration }
        .sorted()
    }

    var intervals: [AIPromptMusicInterval] = []
    for index in 0..<(boundaries.count - 1) {
      let start = boundaries[index]
      let end = boundaries[index + 1]
      guard end - start >= 0.5 else { continue }
      intervals.append(
        AIPromptMusicInterval(
          start: start,
          end: end,
          transition: index == 0 || noveltyAt(start, novelty: novelty, hopSeconds: hopSeconds) > 0.22
            ? "cut"
            : "continue",
          energy: averageValue(normalized, from: start, to: end, hopSeconds: hopSeconds),
          change: noveltyAt(start, novelty: novelty, hopSeconds: hopSeconds)
        )
      )
    }
    if intervals.isEmpty {
      return fallbackAIPromptIntervals(
        totalDuration: totalDuration,
        shotLimitSeconds: shotLimitSeconds
      )
    }
    return intervals
  }

  private func fallbackAIPromptIntervals(
    totalDuration: Double,
    shotLimitSeconds: Double
  ) -> [AIPromptMusicInterval] {
    var intervals: [AIPromptMusicInterval] = []
    var start = 0.0
    var index = 0
    while start < totalDuration - 0.1 {
      let end = min(totalDuration, start + shotLimitSeconds)
      intervals.append(
        AIPromptMusicInterval(
          start: start,
          end: end,
          transition: index == 0 ? "cut" : "continue",
          energy: 0.5,
          change: 0
        )
      )
      start = end
      index += 1
    }
    return intervals
  }

  private func strongestAIPromptBoundary(
    near ideal: Double,
    novelty: [Double],
    hopSeconds: Double,
    minimum: Double,
    maximum: Double
  ) -> Double {
    guard minimum < maximum else { return ideal }
    let startIndex = max(0, Int((minimum / hopSeconds).rounded(.down)))
    let endIndex = min(
      novelty.count - 1,
      Int((maximum / hopSeconds).rounded(.up))
    )
    guard startIndex <= endIndex else {
      return min(max(ideal, minimum), maximum)
    }
    var bestIndex = startIndex
    var bestScore = -Double.infinity
    for index in startIndex...endIndex {
      let time = Double(index) * hopSeconds
      let distancePenalty = abs(time - ideal) / max(0.001, maximum - minimum)
      let score = novelty[index] - distancePenalty * 0.25
      if score > bestScore {
        bestScore = score
        bestIndex = index
      }
    }
    return min(max(Double(bestIndex) * hopSeconds, minimum), maximum)
  }

  private func noveltyAt(
    _ seconds: Double,
    novelty: [Double],
    hopSeconds: Double
  ) -> Double {
    guard !novelty.isEmpty else { return 0 }
    let index = min(
      novelty.count - 1,
      max(0, Int((seconds / hopSeconds).rounded()))
    )
    return novelty[index]
  }

  private func averageValue(
    _ values: [Double],
    from start: Double,
    to end: Double,
    hopSeconds: Double
  ) -> Double {
    guard !values.isEmpty, end > start else { return 0.5 }
    let startIndex = min(
      values.count - 1,
      max(0, Int((start / hopSeconds).rounded(.down)))
    )
    let endIndex = min(
      values.count - 1,
      max(startIndex, Int((end / hopSeconds).rounded(.up)))
    )
    let slice = values[startIndex...endIndex]
    return slice.reduce(0, +) / Double(slice.count)
  }

  private func summarizedAIPromptMusicSections(
    _ intervals: [AIPromptMusicInterval]
  ) -> String {
    guard !intervals.isEmpty else { return "  - none" }
    var sections: [String] = []
    var start = intervals[0].start
    var energyValues: [Double] = []
    for (index, interval) in intervals.enumerated() {
      energyValues.append(interval.energy)
      let shouldClose = index == intervals.count - 1
        || intervals[index + 1].transition == "cut"
        || energyValues.count >= 4
      if shouldClose {
        let averageEnergy = energyValues.reduce(0, +) / Double(energyValues.count)
        sections.append(
          String(
            format: "  - section %02d: %.3f-%.3f sec, average energy %.2f, %@",
            locale: Locale(identifier: "en_US_POSIX"),
            sections.count + 1,
            start,
            interval.end,
            averageEnergy,
            averageEnergy >= 0.67
              ? "high intensity"
              : averageEnergy <= 0.33
                ? "quiet / restrained"
                : "moderate flow"
          )
        )
        if index + 1 < intervals.count {
          start = intervals[index + 1].start
          energyValues = []
        }
      }
    }
    return sections.joined(separator: "\n")
  }

  private func requestAIPrompt(_ request: String) async throws -> String {
    let baseURLString = aiPromptAPIURL.trimmingCharacters(in: .whitespacesAndNewlines)
      .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
    guard let baseURL = URL(string: baseURLString), !baseURLString.isEmpty else {
      throw CocoaError(.fileReadInvalidFileName)
    }
    let model = try await resolvedAIModel(baseURL: baseURL)
    let targetAudioDuration = max(
      0,
      (audioDurationSeconds ?? duration) - audioStartSeconds
    )
    let formattedTargetAudioDuration = String(
      format: "%.3f",
      musicVideoMode ? targetAudioDuration : duration
    )
    let shotLimitSeconds = outputWidth == 1920 && outputHeight == 1080
      ? 6.0
      : min(10.0, max(2.0, duration))
    let formattedShotLimit = String(format: "%.3f", shotLimitSeconds)
    let audioAnalysisSummary = await aiMusicAnalysisSummary(
      targetDuration: targetAudioDuration,
      shotLimitSeconds: shotLimitSeconds
    )
    let subjectReferenceSummary = aiSubjectReferenceSummary()
    let timelineInstruction: String
    if musicVideoMode {
      let manualCuts = musicCutPoints.map {
        String(format: "%.3f", $0.seconds)
      }
      .joined(separator: ", ")
      timelineInstruction = """
        - long_music_video: true
        - target_total_seconds: \(formattedTargetAudioDuration)
        - shot_duration_limit_seconds: \(formattedShotLimit)
        - timeline_requirement: Write a full-song timeline from 0.000 to \(formattedTargetAudioDuration). Do not stop at \(formattedShotLimit) seconds; that value is only the maximum duration of one generated shot.
        - marker_requirement: Every interval marker must include transition kind: [start-end cut] or [start-end continue]. Plain markers like [0.000-3.000] are invalid.
        - first_marker: [0.000-... cut]
        - continuation_rule: Use continue for intervals that physically continue from the previous interval; use cut only for intentional scene/composition changes.
        - last_marker_requirement: The final marker must end exactly at \(formattedTargetAudioDuration) or the full song will be truncated.
        - manual_cut_points_seconds: \(manualCuts.isEmpty ? "none / app may auto-detect cuts from audio" : manualCuts)
        """
    } else {
      timelineInstruction = """
        - long_music_video: false
        - target_total_seconds: \(formattedShotLimit)
        - timeline_requirement: Write one short prompt or a short timeline that ends at \(formattedShotLimit).
        """
    }
    var userText = """
      AIへの指示:
      \(request)

      Current H3 mode context:
      - music_video: \(musicVideoMode ? "true" : "false")
      - audio_mode: \(audioConditioningMode.rawValue)
      - continuation_mode: \(musicVideoContinuationMode.rawValue)
      - resolution: \(outputWidth)x\(outputHeight)
      - duration_or_shot_limit_seconds: \(duration)
      \(timelineInstruction)

      \(subjectReferenceSummary)

      \(audioAnalysisSummary)
      """
    let lyrics = lyricsText.trimmingCharacters(in: .whitespacesAndNewlines)
    if !lyrics.isEmpty {
      userText += """

        Lyrics / song meaning supplied by the user:
        \(lyrics)
        """
    }
    let system = """
      You write MiniMax H3 prompts for mioh upscaler. Return only the finished prompt, no Markdown fences and no explanation.

      For long music videos, use this structure:
      subject_definitions:
      detailed_description:
      GLOBAL CONTINUITY:
      [0.000-... cut]
      [..-.. continue]
      overall_soundscape:
      non_diegetic_music:

      Visual medium rules:
      - Default to live-action, photorealistic, camera-shot video when the user supplies real face/image references or does not explicitly request animation.
      - Choose anime, manga, 2D animation, cartoon, illustration, cel-shading, or animated-film style only when the user's AI instruction asks for that medium or the references are already in that medium.
      - If the user asks for a music video without specifying medium, follow the supplied reference medium and use cinematic live-action MV language for real photos: real camera, practical lighting, lens, natural skin texture, real clothing fabric, believable city locations.
      - Preserve the medium of the supplied references. Real-person face references imply a photoreal live-action person unless the user explicitly asks for an animated reinterpretation.

      Subject reference rules:
      - If SUBJECT REFERENCES are supplied, subject_definitions must define those labels exactly, for example <Subject 1>.
      - Use <Subject 1>, <Subject 2>, etc. in interval bodies whenever the referenced person appears.
      - Do not rename referenced subjects to protagonist, singer, actor, man, woman, boy, girl, or character unless the label also remains present.
      - Reference labels are silent control metadata; never render them as on-screen text.
      - Do not invent extra <Subject N> labels for people who do not have supplied subject references.
      - Reference images apply only to their exact <Subject N>. Other performers, friends, crowds, dancers, reflections, posters, and background people must not inherit or resemble the referenced face.
      - In face-reference mode, <Picture N> means face identity source only. Do not use it as a storyboard image, first frame, pose reference, outfit reference, body reference, background reference, lighting reference, camera-angle reference, crop reference, or composition anchor.
      - Do not write prompts that reproduce the reference photo itself. The referenced subject's clothes, body blocking, pose, environment, framing, lighting, photo mood, and camera angle must come from the interval text, not from the reference picture.
      - When only <Subject 1> is supplied, write other people as unreferenced performers with distinct faces, or keep them as silhouettes, back views, side profiles, motion-blurred crowd, hands, feet, or distant bodies unless the user explicitly needs their face.
      - Avoid close-up face shots of unreferenced people in face-reference mode. If another face must appear near camera, state that it is visually unrelated to the reference pictures and has different facial structure, hairline, eyes, nose, and mouth.

      Critical timeline rules:
      - When long_music_video is true, the prompt must cover the entire target_total_seconds. Do not produce only a 10-second demo unless target_total_seconds is 10.
      - shot_duration_limit_seconds is the maximum length of a single generated shot, not the full music video length.
      - If AUDIO ANALYSIS is supplied, follow its suggested interval table and musical-energy notes. Decide scene cuts, continuation, camera intensity, emotion, and lyric emphasis from that analysis.
      - Every bracket marker must include the transition word: [start-end cut] or [start-end continue]. Plain markers such as [0.000-3.000] are invalid for mioh.
      - Put the interval body on the lines after the marker. Avoid "[start-end cut] body text" on the same line when possible.
      - The first interval should be cut. Later intervals should usually be continue unless the user asks for a clear scene/composition change.
      - The final interval must end at target_total_seconds.

      Use GLOBAL CONTINUITY for stable identity, wardrobe, visual style, music rules, and no-restart/no-duplicate rules. Each flat timeline entry must describe only its local action, camera, prop state, emotion, and next development. Continue entries must not replay the opening action; they continue from the previous physical state.

      Composition discipline for every long music-video interval:
      - Write each interval body as local production notes with these axes in this order: LOCATION, FRAMING, ACTION, CAMERA, optional LIGHTING / COLOR, and optional CONTINUATION NOTE.
      - LOCATION must name the exact place for this interval and may state what previous location should not be visible.
      - FRAMING must specify shot size, lens feel, subject placement, foreground/background geometry, and whether it is wide, medium, close-up, profile, overhead, handheld, etc.
      - ACTION must state what each visible <Subject N> does now, how lyric emotion changes, and what physical state is reached by the interval end.
      - CAMERA must specify motion, speed, screen direction, focus behavior, and whether the camera holds, tracks, circles, pushes in, or cuts away.
      - If several intervals share the same city or character, vary at least three local axes: location, framing, blocking, camera path, foreground objects, background geometry, lighting source, or action endpoint.
      - Do not rely on repeated global mood phrases such as Tokyo neon, cinematic city, actors, rainy night, or emotional connection as the main body of multiple intervals.

      If lyrics are supplied and audio mode is background-music, treat lyrics as meaning, emotion, imagery, and section guidance only. Do not create lip-sync, visible singing mouth shapes, karaoke subtitles, lyric cards, or on-screen text unless explicitly requested. If audio mode is lip-sync, timed lyric lines may guide singing and expression timing.
      """
    let payload = MiniMaxH3AIChatRequest(
      model: model,
      temperature: 0.2,
      maxTokens: 6000,
      messages: [
        .init(role: "system", content: system),
        .init(role: "user", content: userText),
      ]
    )
    var request = URLRequest(url: baseURL.appendingPathComponent("chat/completions"))
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    let apiKey = aiPromptAPIKey.trimmingCharacters(in: .whitespacesAndNewlines)
    if !apiKey.isEmpty {
      request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
    }
    request.httpBody = try JSONEncoder().encode(payload)
    let (data, response) = try await URLSession.shared.data(for: request)
    if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
      let body = String(decoding: data, as: UTF8.self)
      throw NSError(
        domain: "AIPrompt",
        code: http.statusCode,
        userInfo: [NSLocalizedDescriptionKey: body]
      )
    }
    let decoded = try JSONDecoder().decode(MiniMaxH3AIChatResponse.self, from: data)
    let text = (
      decoded.choices?.first?.message?.content
        ?? decoded.choices?.first?.text
        ?? decoded.message?.content
        ?? decoded.response
        ?? ""
    )
      .trimmingCharacters(in: .whitespacesAndNewlines)
    guard !text.isEmpty else {
      throw NSError(
        domain: "AIPrompt",
        code: -1,
        userInfo: [NSLocalizedDescriptionKey: "AI returned an empty prompt"]
      )
    }
    return text
  }

  private func resolvedAIModel(baseURL: URL) async throws -> String {
    let selected = aiPromptModel.trimmingCharacters(in: .whitespacesAndNewlines)
    if !selected.isEmpty, selected != "auto" { return selected }
    var request = URLRequest(url: baseURL.appendingPathComponent("models"))
    let apiKey = aiPromptAPIKey.trimmingCharacters(in: .whitespacesAndNewlines)
    if !apiKey.isEmpty {
      request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
    }
    let (data, response) = try await URLSession.shared.data(for: request)
    if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
      let body = String(decoding: data, as: UTF8.self)
      throw NSError(
        domain: "AIPrompt",
        code: http.statusCode,
        userInfo: [NSLocalizedDescriptionKey: body]
      )
    }
    let decoded = try JSONDecoder().decode(MiniMaxH3AIModelsResponse.self, from: data)
    let models = decoded.data ?? decoded.models ?? []
    if let value = models.compactMap({ $0.id ?? $0.model ?? $0.name }).first,
      !value.isEmpty
    {
      return value
    }
    throw NSError(
      domain: "AIPrompt",
      code: -1,
      userInfo: [NSLocalizedDescriptionKey: "AI server reported no model"]
    )
  }
}

private final class MiniMaxH3TrackingSlider: NSSlider {
  var onTrackingEnded: (() -> Void)?

  override func mouseDown(with event: NSEvent) {
    if isEnabled, bounds.width > 16 {
      let location = convert(event.locationInWindow, from: nil)
      let fraction = min(1, max(0, (location.x - 8) / (bounds.width - 16)))
      doubleValue = minValue + fraction * (maxValue - minValue)
      sendAction(action, to: target)
    }
    super.mouseDown(with: event)
    sendAction(action, to: target)
    onTrackingEnded?()
  }
}

private struct MiniMaxH3AudioScrubSlider: NSViewRepresentable {
  let value: Double
  let maximumValue: Double
  let isEnabled: Bool
  let onChange: (Double) -> Void
  let onTrackingEnded: () -> Void

  final class Coordinator: NSObject {
    var parent: MiniMaxH3AudioScrubSlider

    init(parent: MiniMaxH3AudioScrubSlider) {
      self.parent = parent
    }

    @objc func valueChanged(_ sender: NSSlider) {
      parent.onChange(sender.doubleValue)
    }
  }

  func makeCoordinator() -> Coordinator {
    Coordinator(parent: self)
  }

  func makeNSView(context: Context) -> MiniMaxH3TrackingSlider {
    let slider = MiniMaxH3TrackingSlider(
      value: value,
      minValue: 0,
      maxValue: max(0.001, maximumValue),
      target: context.coordinator,
      action: #selector(Coordinator.valueChanged(_:))
    )
    slider.isContinuous = true
    slider.controlSize = .regular
    slider.setContentHuggingPriority(.defaultLow, for: .horizontal)
    slider.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
    slider.setAccessibilityLabel("構図変更位置の試聴")
    slider.onTrackingEnded = { [weak coordinator = context.coordinator] in
      coordinator?.parent.onTrackingEnded()
    }
    return slider
  }

  func updateNSView(
    _ slider: MiniMaxH3TrackingSlider,
    context: Context
  ) {
    context.coordinator.parent = self
    slider.minValue = 0
    slider.maxValue = max(0.001, maximumValue)
    slider.isEnabled = isEnabled && maximumValue > 0
    let clamped = min(max(0, value), max(0, maximumValue))
    if abs(slider.doubleValue - clamped) > 1e-6 {
      slider.doubleValue = clamped
    }
    slider.setAccessibilityValue(
      MiniMaxH3MusicCutTimeline.clock(clamped)
    )
  }

  func sizeThatFits(
    _ proposal: ProposedViewSize,
    nsView: MiniMaxH3TrackingSlider,
    context: Context
  ) -> CGSize? {
    CGSize(width: proposal.width ?? 480, height: 24)
  }
}

private struct MiniMaxH3MusicCutTimeline: View {
  let duration: Double
  let points: [MiniMaxH3MusicCutPoint]
  let previewSeconds: Double?
  let selectedID: UUID?
  let isDisabled: Bool
  let onMove: (UUID, Double) -> Void
  let onSelect: (UUID) -> Void
  let onSetPreview: (Double, Bool) -> Void
  let onFinishPreview: () -> Void
  let onPreview: (Double, Bool) -> Void
  let onStopPreview: () -> Void

  private let coordinateSpaceName = "MiniMaxH3MusicCutTimeline"

  var body: some View {
    GeometryReader { geometry in
      let inset = 9.0
      let usableWidth = max(1, geometry.size.width - 2 * inset)
      ZStack {
        MiniMaxH3AudioScrubSlider(
          value: min(max(0, previewSeconds ?? 0), max(0, duration)),
          maximumValue: duration,
          isEnabled: !isDisabled,
          onChange: { onSetPreview($0, false) },
          onTrackingEnded: onFinishPreview
        )
          .frame(maxWidth: .infinity)
        ForEach(points) { point in
          let fraction = duration > 0
            ? min(1, max(0, point.seconds / duration))
            : 0
          let x = inset + fraction * usableWidth
          VStack(spacing: 1) {
            Text(Self.clock(point.seconds))
              .font(.caption2.monospacedDigit())
              .padding(.horizontal, 4)
              .padding(.vertical, 1)
              .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 4))
            Rectangle()
              .fill(Color.orange)
              .frame(width: 2, height: 16)
            Circle()
              .fill(Color.orange)
              .frame(width: 12, height: 12)
              .overlay {
                if selectedID == point.id {
                  Circle().stroke(Color.primary, lineWidth: 1.5)
                }
              }
          }
          .position(x: x, y: geometry.size.height / 2)
          .contentShape(Rectangle())
          .onTapGesture {
            onSelect(point.id)
            onPreview(point.seconds, true)
          }
          .gesture(
            DragGesture(
              minimumDistance: 0,
              coordinateSpace: .named(coordinateSpaceName)
            )
            .onChanged { value in
              guard !isDisabled, duration > 0 else { return }
              onSelect(point.id)
              let fraction = min(
                1,
                max(0, (value.location.x - inset) / usableWidth)
              )
              let seconds = fraction * duration
              onMove(point.id, seconds)
              onPreview(seconds, false)
            }
            .onEnded { value in
              guard !isDisabled, duration > 0 else { return }
              let fraction = min(
                1,
                max(0, (value.location.x - inset) / usableWidth)
              )
              onMove(point.id, fraction * duration)
              onStopPreview()
            }
          )
          .help("構図変更 \(Self.clock(point.seconds))")
        }
      }
      .frame(
        width: max(1, geometry.size.width),
        height: max(1, geometry.size.height)
      )
      .coordinateSpace(name: coordinateSpaceName)
    }
    .frame(maxWidth: .infinity, minHeight: 50, maxHeight: 50)
  }

  static func clock(_ seconds: Double) -> String {
    let rounded = max(0, Int(seconds.rounded()))
    return String(format: "%d:%02d", rounded / 60, rounded % 60)
  }
}

struct MiniMaxH3GenerationView: View {
  @ObservedObject var controller: MiniMaxH3Controller
  let upscalerInputURL: URL?

  var body: some View {
    Form {
      Section("動画生成（MiniMax H3）") {
        LabeledContent("生成モード") {
          Picker(
            "",
            selection: Binding(
              get: { controller.supportsPromptOnly },
              set: { controller.selectGenerationMode(promptOnly: $0) }
            )
          ) {
            Text("参照画像／動画").tag(false)
            Text("プロンプトのみ").tag(true)
          }
          .labelsHidden()
          .pickerStyle(.segmented)
          .frame(width: 260)
        }
        LabeledContent("入力") {
          if controller.supportsPromptOnly {
            Text(controller.inputSummary)
              .foregroundStyle(.secondary)
          } else {
            HStack {
              Text(controller.inputSummary)
                .lineLimit(1)
                .truncationMode(.middle)
                .textSelection(.enabled)
              Button("選択", action: controller.chooseInput)
              if !controller.usesUpscalerInput, upscalerInputURL != nil {
                Button("アップスケール入力を使用") {
                  controller.useUpscalerInput(upscalerInputURL)
                }
              }
            }
          }
        }
        if controller.isImageSequence {
          LabeledContent("画像の参照範囲") {
            Picker(
              "",
              selection: Binding(
                get: { controller.imageReferenceScope },
                set: { controller.selectImageReferenceScope($0) }
              )
            ) {
              ForEach(MiniMaxH3ImageReferenceScope.allCases) { scope in
                Text(scope.label).tag(scope)
              }
            }
            .labelsHidden()
            .pickerStyle(.segmented)
            .frame(width: 220)
            .disabled(controller.isRunning)
          }
          if controller.imageReferenceScope == .faceOnly {
            LabeledContent("顔参照") {
              HStack(spacing: 10) {
                if controller.isDetectingFaces {
                  ProgressView().controlSize(.small)
                  Text("検出中")
                } else {
                  Text(
                    "使用 \(controller.selectedFaceReferenceCount) / "
                      + "検出 \(controller.faceReferences.count)"
                  )
                    .monospacedDigit()
                  Button("再検出", action: controller.detectFaces)
                    .disabled(controller.isRunning)
                  Button(
                    "選択顔を<Subject 1>へまとめる",
                    action: controller.groupSelectedFacesAsOneSubject
                  )
                    .disabled(
                      controller.isRunning
                        || controller.selectedFaceReferenceCount == 0
                    )
                }
              }
            }
            if !controller.faceReferences.isEmpty {
              DisclosureGroup(
                "検出した顔（使用する顔とSubject番号を指定）"
              ) {
                VStack(alignment: .leading, spacing: 8) {
                  ForEach(controller.faceReferences) { reference in
                    MiniMaxH3FaceReferenceRow(
                      controller: controller,
                      reference: reference
                    )
                    if reference.id != controller.faceReferences.last?.id {
                      Divider()
                    }
                  }
                }
                .padding(.vertical, 4)
              }
            }
            Text(
              "同一人物の別画像には同じSubject番号を指定してください。プロンプトでは<Subject 1>の形式で指定します。H3には選択した顔クロップだけを渡し、服装・姿勢・背景・構図はプロンプトから生成します。"
            )
              .font(.caption)
              .foregroundStyle(.secondary)
          }
        }
        if controller.isImageSequence, controller.inputURLs.count > 1 {
          DisclosureGroup("選択した画像（\(controller.inputURLs.count)枚）") {
            ForEach(controller.inputURLs, id: \.path) { url in
              Text(url.path)
                .font(.caption.monospaced())
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
          }
        }
        if controller.isImageSequence {
          LabeledContent("音源") {
            HStack {
              Text(controller.audioInputSummary)
                .lineLimit(1)
                .truncationMode(.middle)
                .textSelection(.enabled)
              Button("選択", action: controller.chooseAudioInput)
              if controller.audioInputURL != nil {
                Button("解除", action: controller.clearAudioInput)
              }
            }
          }
          if controller.audioInputURL != nil {
            LabeledContent("音源の使い方") {
              Picker("", selection: $controller.audioConditioningMode) {
                ForEach(MiniMaxH3AudioConditioningMode.allCases, id: \.self) {
                  mode in
                  Text(mode.label).tag(mode)
                }
              }
              .labelsHidden()
              .pickerStyle(.segmented)
              .frame(width: 330)
              .disabled(controller.isRunning)
            }
            LabeledContent("音源開始位置") {
              HStack {
                TextField(
                  "",
                  value: $controller.audioStartSeconds,
                  format: .number.precision(.fractionLength(2))
                )
                .multilineTextAlignment(.trailing)
                .frame(width: 90)
                Text("秒")
                  .foregroundStyle(.secondary)
              }
            }
            Text(
              controller.audioConditioningMode.helpText
                + " 元音源を音声latentとして生成中も固定し、完成動画にも同じ音源をそのまま使用します。1ショットは最大10秒です。"
            )
              .font(.caption)
              .foregroundStyle(.secondary)
            DisclosureGroup("歌詞・曲の意味") {
              VStack(alignment: .leading, spacing: 8) {
                LabeledContent("検索") {
                  HStack {
                    TextField(
                      "曲名 アーティスト",
                      text: $controller.lyricsSearchQuery
                    )
                    .disabled(controller.isRunning)
                    Button("ブラウザで検索", action: controller.openLyricsSearch)
                      .disabled(controller.isRunning)
                  }
                }
                HStack {
                  Button("歌詞ファイルを読み込む", action: controller.chooseLyricsFile)
                    .disabled(controller.isRunning)
                  if !controller.lyricsText.isEmpty {
                    Button("歌詞をクリア") {
                      controller.lyricsText = ""
                    }
                    .disabled(controller.isRunning)
                  }
                }
                TextEditor(text: $controller.lyricsText)
                  .font(.system(.caption, design: .monospaced))
                  .frame(minHeight: 120)
                  .disabled(controller.isRunning)
                  .overlay(
                    RoundedRectangle(cornerRadius: 6)
                      .stroke(.separator.opacity(0.35))
                  )
                Text(
                  controller.audioConditioningMode == .lipSync
                    ? "タイムコード付き歌詞（LRC形式など）を貼ると、歌唱や表情タイミングの参考としてプロンプトに渡します。"
                    : "BGM参照では、貼り付けた歌詞は意味・感情・曲構成の参考として渡し、口パクや字幕は明示しない限り生成しません。検索はブラウザを開くだけで、自動取得はしません。"
                )
                  .font(.caption)
                  .foregroundStyle(.secondary)
              }
              .padding(.vertical, 4)
            }
            Toggle("長尺Music Videoとして音源の最後まで連続生成", isOn: $controller.musicVideoMode)
              .disabled(controller.isRunning)
            if controller.musicVideoMode {
              Text(controller.musicVideoSummary)
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
              LabeledContent("Part継続") {
                Picker("", selection: $controller.musicVideoContinuationMode) {
                  Text("Hybrid AV（推奨）")
                    .tag(MiniMaxH3MusicVideoContinuationMode.hybridAV)
                  Text("latent-prefix（従来方式）")
                    .tag(MiniMaxH3MusicVideoContinuationMode.latentPrefix)
                  Text("Firstのみ（高速）")
                    .tag(MiniMaxH3MusicVideoContinuationMode.firstFrame)
                  Text("First＋Codex指定Last（1パス）")
                    .tag(
                      MiniMaxH3MusicVideoContinuationMode.firstAndProvidedLast
                    )
                  Text("First＋mioh生成Last（全自動）")
                    .tag(
                      MiniMaxH3MusicVideoContinuationMode.firstAndGeneratedLast
                    )
                }
                .labelsHidden()
                .pickerStyle(.menu)
                .disabled(controller.isRunning)
              }
              Text(
                controller.musicVideoContinuationMode == .hybridAV
                  ? "直近22フレームの映像・音声latentを完全固定し、その直前2トークンを履歴として参照します。ContinuumとH3-Extendを重複なしで融合した方式です。"
                  : controller.musicVideoContinuationMode == .latentPrefix
                  ? "前Part末尾の映像latentを次Partの固定prefixとして渡し、重複する22フレームを出力から除く互換方式です。"
                  : controller.musicVideoContinuationMode == .firstAndProvidedLast
                  ? "Codexなどで作成したPart終端画像をLast条件にして本生成します。"
                  : controller.musicVideoContinuationMode == .firstAndGeneratedLast
                    ? "miohがFirst-onlyドラフトの最終フレームを作り、Last条件にして本生成します。"
                    : "前Partの最終フレームだけをFirst条件として続きPartを生成します。"
              )
                .font(.caption)
                .foregroundStyle(.secondary)
              if controller.musicVideoContinuationMode == .firstAndProvidedLast {
                LabeledContent("Last frameフォルダ") {
                  HStack {
                    Text(controller.musicVideoLastFrameDirectory.isEmpty
                      ? "未指定"
                      : controller.musicVideoLastFrameDirectory)
                      .lineLimit(1)
                      .truncationMode(.middle)
                      .textSelection(.enabled)
                    Button(
                      "選択",
                      action: controller.chooseMusicVideoLastFrameDirectory
                    )
                    .disabled(controller.isRunning)
                  }
                }
                Text("画像名は shot-0000-part-01-last.png の形式です。")
                  .font(.caption)
                  .foregroundStyle(.secondary)
              }
              GroupBox("構図変更ポイント") {
                VStack(alignment: .leading, spacing: 8) {
                  Picker("", selection: $controller.usesAutomaticMusicCuts) {
                    Text("音源から自動検出").tag(true)
                    Text("スライダーで手動指定").tag(false)
                  }
                  .labelsHidden()
                  .pickerStyle(.segmented)
                  .disabled(controller.isRunning)

                  if !controller.usesAutomaticMusicCuts {
                    MiniMaxH3MusicCutTimeline(
                      duration: controller.availableMusicDuration,
                      points: controller.musicCutPoints,
                      previewSeconds: controller.musicPreviewSeconds,
                      selectedID: controller.selectedMusicCutPointID,
                      isDisabled: controller.isRunning,
                      onMove: controller.moveMusicCutPoint,
                      onSelect: controller.selectMusicCutPoint,
                      onSetPreview: controller.setMusicPreview,
                      onFinishPreview: controller.finishMusicPreview,
                      onPreview: controller.previewMusic,
                      onStopPreview: controller.stopAudioPreview
                    )
                    HStack {
                      Text("0:00")
                      Spacer()
                      Text(
                        MiniMaxH3MusicCutTimeline.clock(
                          controller.availableMusicDuration
                        )
                      )
                    }
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)

                    HStack {
                      Text(
                        controller.musicPreviewSeconds == nil
                          ? "線上をクリックして音を確認し、選択でポイントを確定"
                          : "青は試聴位置、橙は確定済みポイント"
                      )
                        .font(.caption)
                        .foregroundStyle(.secondary)
                      Spacer()
                      if let previewSeconds = controller.musicPreviewSeconds,
                        !controller.musicPreviewIsCommitted
                      {
                        Text(
                          MiniMaxH3MusicCutTimeline.clock(previewSeconds)
                        )
                        .font(.caption.monospacedDigit())
                        Button(
                          "選択",
                          action: controller.commitMusicPreviewPoint
                        )
                        .disabled(
                          controller.isRunning
                            || !controller.canCommitMusicPreviewPoint
                        )
                      }
                      if (controller.musicPreviewSeconds == nil
                          || controller.musicPreviewIsCommitted),
                        let selectedID = controller.selectedMusicCutPointID,
                        let seconds = controller.selectedMusicCutPointSeconds
                      {
                        Text("確定ポイント")
                          .font(.caption)
                        TextField(
                          "",
                          value: Binding(
                            get: { seconds },
                            set: {
                              controller.moveMusicCutPoint(
                                id: selectedID,
                                to: $0
                              )
                              controller.previewMusic(
                                at: $0,
                                automaticallyStop: true
                              )
                            }
                          ),
                          format: .number.precision(.fractionLength(2))
                        )
                        .multilineTextAlignment(.trailing)
                        .frame(width: 76)
                        Text("秒")
                          .foregroundStyle(.secondary)
                        Button(
                          "削除",
                          action: controller.removeSelectedMusicCutPoint
                        )
                        .disabled(controller.isRunning)
                      }
                      Button("全消去", action: controller.clearMusicCutPoints)
                        .disabled(controller.isRunning || controller.musicCutPoints.isEmpty)
                    }
                    Text(
                      "ポイントは音源開始位置からの相対時間です。前後2秒以上あけて指定します。選択後は次の位置をクリックしてください。ポイントなしなら全曲を同じ構図で継続します。"
                    )
                    .font(.caption)
                    .foregroundStyle(.secondary)
                  }
                }
                .padding(.top, 2)
                .frame(maxWidth: .infinity, alignment: .leading)
              }
              .frame(maxWidth: .infinity)
            }
          }
        }
        Text(
          controller.supportsPromptOnly
            ? "FL2VAモデルが文字条件だけから映像と音声を生成します。参照画像・動画は使用しません。"
            : "動画は1本、画像は最大8枚を選択できます。参照素材は縦横比を保って中央に収め、生成動画の縦横比は下の解像度で指定します。"
        )
          .font(.caption)
          .foregroundStyle(.secondary)
        LabeledContent("実行方式") {
          Text("Core AI（BF16 DiT はGPU、対応部分はANE）")
        }
        LabeledContent("モデルマニフェスト") {
          HStack {
            TextField("", text: $controller.manifestPath)
              .textFieldStyle(.roundedBorder)
            Button(action: controller.chooseManifest) {
              Image(systemName: "folder")
            }
            .buttonStyle(.borderless)
          }
        }
        LabeledContent("Core AIキャッシュ") {
          VStack(alignment: .trailing, spacing: 6) {
            HStack {
              TextField("システム標準", text: $controller.coreAICacheRoot)
                .textFieldStyle(.roundedBorder)
              Button(action: controller.chooseCoreAICacheRoot) {
                Image(systemName: "folder")
              }
              .buttonStyle(.borderless)
              Button("標準", action: controller.useSystemCoreAICache)
                .disabled(controller.coreAICacheRoot.isEmpty || controller.isRunning)
              Button("削除", role: .destructive, action: controller.clearCoreAICache)
                .disabled(controller.coreAICacheRoot.isEmpty || controller.isRunning)
            }
            Text(controller.coreAICacheSummary)
              .font(.caption)
              .foregroundStyle(.secondary)
              .lineLimit(2)
              .textSelection(.enabled)
          }
        }
        Text("macOS 27.2以降では指定先にCore AI特殊化キャッシュを保存します。モデル本体と生成途中キャッシュは移動しません。")
          .font(.caption)
          .foregroundStyle(.secondary)
        if !controller.supportsRuntime {
          Text("Core AI版MiniMax H3にはmacOS 27以降が必要です")
            .foregroundStyle(.red)
        } else if !controller.modelReady {
          Text("モデルは内蔵されません。外部のMiniMax H3 manifest.jsonを選択してください")
            .foregroundStyle(.orange)
        }
      }
      Section("AIプロンプト生成") {
        VStack(alignment: .leading, spacing: 6) {
          Text("AIへの指示")
            .font(.caption)
            .foregroundStyle(.secondary)
          TextEditor(text: $controller.aiPromptRequest)
            .font(.body)
            .frame(minHeight: 72, maxHeight: 140)
            .overlay(
              RoundedRectangle(cornerRadius: 6)
                .stroke(.separator.opacity(0.35))
            )
        }
        VStack(alignment: .leading, spacing: 8) {
          HStack(alignment: .top, spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
              Text("AI種別")
                .font(.caption)
                .foregroundStyle(.secondary)
              Picker("", selection: $controller.aiPromptProvider) {
                ForEach(MiniMaxH3AIPromptProvider.allCases) { provider in
                  Text(provider.label).tag(provider)
                }
              }
              .labelsHidden()
              .frame(width: 180)
              .onChange(of: controller.aiPromptProvider) { _, _ in
                controller.applyAIPromptProviderDefaults()
              }
            }
            VStack(alignment: .leading, spacing: 4) {
              Text("モデル")
                .font(.caption)
                .foregroundStyle(.secondary)
              TextField("auto", text: $controller.aiPromptModel)
                .textFieldStyle(.roundedBorder)
                .frame(width: 120)
            }
            Spacer(minLength: 8)
            Button(
              controller.isGeneratingAIPrompt ? "生成中…" : "AIプロンプト生成",
              action: controller.generateAIPrompt
            )
            .disabled(controller.isGeneratingAIPrompt || controller.isRunning)
            .padding(.top, 18)
          }
          VStack(alignment: .leading, spacing: 4) {
            Text("API URL")
              .font(.caption)
              .foregroundStyle(.secondary)
            TextField("", text: $controller.aiPromptAPIURL)
              .textFieldStyle(.roundedBorder)
          }
          VStack(alignment: .leading, spacing: 4) {
            Text("APIキー")
              .font(.caption)
              .foregroundStyle(.secondary)
            SecureField("必要な場合のみ", text: $controller.aiPromptAPIKey)
              .textFieldStyle(.roundedBorder)
            Text("ローカルAIでは通常空欄でOK")
              .font(.caption)
              .foregroundStyle(.secondary)
          }
        }
        VStack(alignment: .leading, spacing: 6) {
          HStack {
            Text("AI生成プロンプト")
              .font(.caption)
              .foregroundStyle(.secondary)
            Spacer()
            Button("MiniMaxプロンプトへ反映", action: controller.applyGeneratedAIPrompt)
              .disabled(
                controller.aiGeneratedPrompt
                  .trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                  || controller.isRunning
              )
          }
          TextEditor(text: $controller.aiGeneratedPrompt)
            .font(.body)
            .frame(minHeight: 96, maxHeight: 180)
            .overlay(
              RoundedRectangle(cornerRadius: 6)
                .stroke(.separator.opacity(0.35))
            )
        }
        Text(
          "OpenAI互換APIとして、ローカルGemma / Ollama / LM Studio / カスタムURLにSwiftから直接送信します。歌詞・曲の意味欄に入力があれば、AIプロンプト生成にも自動で含めます。"
        )
          .font(.caption)
          .foregroundStyle(.secondary)
      }
      Section("プロンプト") {
        TextEditor(text: $controller.prompt)
          .font(.body)
          .frame(minHeight: 64, maxHeight: 120)
        Text(
          controller.supportsPromptOnly
            ? "自由入力（最大4152 Qwenトークン。超過分は末尾を省略）"
            : "自由入力（参照画像・動画の視覚トークンを除く範囲を使用。超過分は末尾を省略）"
        )
          .font(.caption)
          .foregroundStyle(.secondary)
      }
      Section("生成設定") {
        LabeledContent("解像度") {
          HStack(spacing: 10) {
            Picker("", selection: $controller.resolutionProfileID) {
              ForEach(MiniMaxH3Controller.resolutionProfiles) { profile in
                Text(profile.label).tag(profile.id)
              }
            }
            .labelsHidden()
            .frame(width: 230)
            Text("24fps固定")
              .font(.caption)
              .foregroundStyle(.secondary)
              .monospacedDigit()
          }
        }
        LabeledContent("長さ") {
          if controller.isDurationFixed {
            Text("6.0秒（1080p固定）")
              .monospacedDigit()
          } else {
            HStack(spacing: 10) {
              Slider(
                value: $controller.duration,
                in: 2...controller.maximumShotDuration,
                step: 0.5
              )
                .frame(width: 180)
              TextField(
                "",
                value: $controller.duration,
                format: .number.precision(.fractionLength(1))
              )
              .multilineTextAlignment(.trailing)
              .frame(width: 54)
              Text("秒")
                .foregroundStyle(.secondary)
            }
          }
        }
        Text(
          controller.audioInputURL == nil
            ? "1080pは6秒固定、その他は2〜15秒。高解像度ほど処理時間とメモリ使用量が増えます"
            : "音源使用時は1ショット2〜10秒（1080pは6秒固定）です"
        )
          .font(.caption)
          .foregroundStyle(.secondary)
        LabeledContent("Seed") {
          TextField("", text: $controller.seed)
            .multilineTextAlignment(.trailing)
            .frame(width: 220)
        }
        LabeledContent("出力") {
          HStack {
            TextField("", text: $controller.outputPath)
              .textFieldStyle(.roundedBorder)
            Button(action: controller.chooseOutput) {
              Image(systemName: "folder")
            }
            .buttonStyle(.borderless)
          }
        }
      }
      Section("進捗") {
        ProgressView(value: controller.progress)
        Text(controller.status)
          .font(.callout.monospacedDigit())
        if !controller.musicAnalysisSummary.isEmpty {
          LabeledContent("音源解析") {
            Text(controller.musicAnalysisSummary)
              .font(.caption.monospacedDigit())
              .multilineTextAlignment(.trailing)
              .textSelection(.enabled)
          }
        }
        ScrollView {
          Text(controller.log.isEmpty ? " " : controller.log)
            .font(.system(.caption, design: .monospaced))
            .textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .topLeading)
        }
        .frame(minHeight: 110, maxHeight: 180)
        .background(Color(nsColor: .textBackgroundColor))
      }
    }
    .formStyle(.grouped)
    .onAppear { controller.prepare(upscalerInput: upscalerInputURL) }
    .onChange(of: upscalerInputURL) { _, value in
      controller.prepare(upscalerInput: value)
    }
    .onChange(of: controller.audioStartSeconds) { _, _ in
      controller.audioStartDidChange()
    }
  }
}

private struct MiniMaxH3FaceReferenceRow: View {
  @ObservedObject var controller: MiniMaxH3Controller
  let reference: MiniMaxH3FaceReference

  var body: some View {
    HStack(spacing: 10) {
      Toggle(
        "",
        isOn: Binding(
          get: { reference.isSelected },
          set: {
            controller.setFaceReferenceSelected(reference.id, selected: $0)
          }
        )
      )
        .labelsHidden()
        .disabled(controller.isRunning)
      if let preview = NSImage(contentsOf: reference.cropURL) {
        Image(nsImage: preview)
          .resizable()
          .scaledToFill()
          .frame(width: 58, height: 58)
          .clipShape(RoundedRectangle(cornerRadius: 6))
      } else {
        Image(systemName: "person.crop.square")
          .frame(width: 58, height: 58)
          .background(.quaternary)
          .clipShape(RoundedRectangle(cornerRadius: 6))
      }
      VStack(alignment: .leading, spacing: 3) {
        Text(reference.sourceLabel)
          .lineLimit(1)
          .truncationMode(.middle)
        Text("検出信頼度 \(Int((reference.confidence * 100).rounded()))%")
          .font(.caption.monospacedDigit())
          .foregroundStyle(.secondary)
      }
      Spacer()
      Picker(
        "Subject",
        selection: Binding(
          get: { reference.subjectIndex },
          set: {
            controller.setFaceReferenceSubject(
              reference.id,
              subjectIndex: $0
            )
          }
        )
      ) {
        ForEach(1...8, id: \.self) { index in
          Text("<Subject \(index)>").tag(index)
        }
      }
        .labelsHidden()
        .pickerStyle(.menu)
        .frame(width: 140, alignment: .trailing)
        .layoutPriority(1)
        .accessibilityLabel("Subject")
        .accessibilityValue("<Subject \(reference.subjectIndex)>")
        .disabled(controller.isRunning || !reference.isSelected)
    }
  }
}
