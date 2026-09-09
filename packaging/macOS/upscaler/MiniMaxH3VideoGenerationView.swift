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

  @Published var prompt = "モザイクを除去して最高品質の動画を生成する。"
  @Published fileprivate var promptAssistantProvider:
    MiniMaxH3PromptAssistantProvider = .llamaCpp
  @Published var promptAssistantEndpoint = "http://127.0.0.1:18080"
  @Published var promptAssistantModel = "gemma-4"
  @Published var promptAssistantInstruction = ""
  @Published var promptAssistantOutput = ""
  @Published private(set) var isGeneratingPrompt = false
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
      refreshQwenSequenceLength()
    }
  }
  @Published private(set) var supportsPromptOnly = false
  @Published private(set) var qwenSequenceLength = 4152
  @Published private(set) var inputURLs: [URL] = []
  @Published private(set) var audioInputURL: URL?
  @Published var audioStartSeconds = 0.0
  @Published private(set) var audioDurationSeconds: Double?
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
  @Published fileprivate var referenceEditMode: H3ReferenceEditMode = .none
  @Published var referenceEditTargetDescription = ""
  @Published private(set) var videoMaskCandidates: [MiniMaxH3VideoMaskCandidate] = []
  @Published private(set) var selectedVideoMaskCandidateID: String?
  @Published private(set) var isDetectingVideoMaskCandidates = false
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
  private var videoMaskCandidateTask: Task<Void, Never>?
  private var videoMaskCandidateDirectory: URL?
  private var promptAssistantTask: Task<Void, Never>?

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
    refreshConditioningMode()
    refreshQwenSequenceLength()
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

  var inputImageURLs: [URL] {
    inputURLs.filter(Self.isImage)
  }

  var inputVideoURL: URL? {
    inputURLs.first(where: Self.isMovie)
  }

  var inputSummary: String {
    if supportsPromptOnly { return "なし（プロンプトのみ）" }
    if inputURLs.isEmpty { return "未指定" }
    let imageURLs = inputImageURLs
    if !imageURLs.isEmpty {
      if imageReferenceScope == .faceOnly {
        if isDetectingFaces { return "Subject候補の顔を検出中" }
        let prefix = inputVideoURL == nil ? "" : "動画 + "
        return "\(prefix)顔参照 \(selectedFaceReferences.count)件（検出\(faceReferences.count)件）"
      }
      if inputVideoURL == nil, imageURLs.count == 1 { return imageURLs[0].path }
      let prefix = inputVideoURL == nil ? "" : "動画 + "
      return "\(prefix)Subject参照画像 \(imageURLs.count)枚（先頭画像が基準）"
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
    if referenceEditMode != .none {
      guard inputVideoURL != nil, !inputImageURLs.isEmpty,
        !isDetectingVideoMaskCandidates
      else { return false }
    }
    if audioInputURL != nil {
      guard (!inputImageURLs.isEmpty || inputVideoURL != nil),
        duration <= Self.maximumAudioConditioningDuration
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

  func generatePromptWithLocalAI() {
    guard !isRunning, !isGeneratingPrompt else { return }
    promptAssistantTask?.cancel()
    isGeneratingPrompt = true
    status = "ローカルAIでH3プロンプトを生成中"
    let instruction = promptAssistantInstruction
    let provider = promptAssistantProvider
    let endpoint = promptAssistantEndpoint
    let model = promptAssistantModel
    let context = MiniMaxH3PromptAssistantContext(
      promptOnly: supportsPromptOnly,
      hasVideo: inputVideoURL != nil,
      imageCount: inputImageURLs.count,
      hasAudio: audioInputURL != nil,
      faceOnly: imageReferenceScope == .faceOnly,
      referenceEditMode: referenceEditMode,
      durationSeconds: duration,
      qwenSequenceLength: qwenSequenceLength
    )
    promptAssistantTask = Task { [weak self] in
      do {
        let generated = try await MiniMaxH3PromptAssistant.generatePrompt(
          instruction: instruction,
          provider: provider,
          endpoint: endpoint,
          model: model,
          context: context
        )
        guard !Task.isCancelled else { return }
        await MainActor.run {
          guard let self else { return }
          self.promptAssistantOutput = generated
          self.isGeneratingPrompt = false
          self.status = "H3プロンプトを生成しました"
        }
      } catch {
        guard !Task.isCancelled else { return }
        await MainActor.run {
          guard let self else { return }
          self.isGeneratingPrompt = false
          self.status = "H3プロンプト生成に失敗しました"
          self.appendLog("promptAssistant: \(error.localizedDescription)\n")
        }
      }
    }
  }

  func applyGeneratedPrompt() {
    guard !isRunning else { return }
    let generated = promptAssistantOutput.trimmingCharacters(
      in: .whitespacesAndNewlines
    )
    guard !generated.isEmpty else { return }
    prompt = generated
    status = "生成したH3プロンプトを反映しました"
  }

  func setReferenceEditMode(_ mode: H3ReferenceEditMode, enabled: Bool) {
    guard !isRunning else { return }
    referenceEditMode = enabled ? mode : .none
    if referenceEditMode != .none, supportsPromptOnly {
      selectGenerationMode(promptOnly: false)
    }
    if referenceEditMode != .none {
      detectVideoMaskCandidates()
    }
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
    panel.title = "リップシンク用の音源を選択"
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
        self.status = "外部音源をリップシンク条件に使用します"
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
    musicVideoMode = false
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
    let imageURLs = urls.filter(Self.isImage)
    let videoURLs = urls.filter(Self.isMovie)
    if videoURLs.count > 1 || imageURLs.count + videoURLs.count != urls.count {
      status = "動画は1本、画像は最大\(Self.maximumIdentityImages)枚まで同時選択できます"
      return
    }
    if imageURLs.count > Self.maximumIdentityImages {
      status = "Subject参照画像は最大\(Self.maximumIdentityImages)枚です"
      return
    }
    if !imageURLs.isEmpty {
      let sortedImages = imageURLs.sorted {
        $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent)
          == .orderedAscending
      }
      urls = videoURLs + sortedImages
    }
    setInputURLs(urls, upscalerInput: false)
  }

  private func setInputURLs(_ urls: [URL], upscalerInput: Bool) {
    resetFaceReferences()
    resetVideoMaskCandidates()
    inputURLs = urls
    usesUpscalerInput = upscalerInput
    if inputVideoURL == nil || inputImageURLs.isEmpty {
      referenceEditMode = .none
    }
    guard let first = urls.first else { return }
    let suffix = urls.count > 1 ? "-\(urls.count)-images" : ""
    let proposed = first.deletingPathExtension().path
      + suffix + "-minimax-h3.mp4"
    if outputPath.isEmpty || outputPath == automaticOutputPath {
      outputPath = proposed
      automaticOutputPath = proposed
    }
    if imageReferenceScope == .faceOnly, !inputImageURLs.isEmpty {
      detectFaces()
    }
    if referenceEditMode != .none {
      detectVideoMaskCandidates()
    }
  }

  func detectVideoMaskCandidates() {
    guard !isRunning, referenceEditMode != .none,
      let videoURL = inputVideoURL,
      !inputImageURLs.isEmpty
    else { return }
    videoMaskCandidateTask?.cancel()
    let directory = FileManager.default.temporaryDirectory
      .appendingPathComponent(
        "mioh-h3-video-mask-candidates-\(UUID().uuidString)",
        isDirectory: true
      )
    videoMaskCandidateDirectory = directory
    isDetectingVideoMaskCandidates = true
    videoMaskCandidates = []
    selectedVideoMaskCandidateID = nil
    status = "動画内の人物候補を検出中"
    videoMaskCandidateTask = Task { [weak self] in
      do {
        let candidates = try await MiniMaxH3ReferenceVideoMaskProcessor
          .detectMaskCandidates(
            sourceURL: videoURL,
            destinationDirectory: directory
          )
        guard !Task.isCancelled else { return }
        await MainActor.run {
          guard let self, self.inputVideoURL == videoURL else { return }
          self.videoMaskCandidates = candidates
          self.selectedVideoMaskCandidateID = candidates.first?.id
          self.isDetectingVideoMaskCandidates = false
          self.status = candidates.isEmpty
            ? "動画内の人物候補を検出できませんでした"
            : "動画内の人物候補を\(candidates.count)件検出しました"
        }
      } catch {
        guard !Task.isCancelled else { return }
        await MainActor.run {
          guard let self, self.inputVideoURL == videoURL else { return }
          self.isDetectingVideoMaskCandidates = false
          self.status = "動画内の人物候補検出に失敗しました"
          self.appendLog("人物候補: \(error.localizedDescription)\n")
        }
      }
    }
  }

  func selectVideoMaskCandidate(_ id: String) {
    guard !isRunning else { return }
    selectedVideoMaskCandidateID = id
  }

  private var selectedVideoMaskCandidate: MiniMaxH3VideoMaskCandidate? {
    guard let selectedVideoMaskCandidateID else { return nil }
    return videoMaskCandidates.first { $0.id == selectedVideoMaskCandidateID }
  }

  private var selectedVideoMaskCandidateIndex: Int? {
    selectedVideoMaskCandidate?.index
  }

  private var validInputSelection: Bool {
    if supportsPromptOnly { return inputURLs.isEmpty }
    guard !inputURLs.isEmpty else { return false }
    let imageURLs = inputImageURLs
    let videoCount = inputVideoURL == nil ? 0 : 1
    if !imageURLs.isEmpty, imageReferenceScope == .faceOnly {
      let count = selectedFaceReferences.count
      return !isDetectingFaces && count > 0
        && count <= Self.maximumIdentityImages
    }
    return videoCount + imageURLs.count == inputURLs.count
      && videoCount <= 1
      && imageURLs.count <= Self.maximumIdentityImages
  }

  private func promptWithReferenceEditInstructions(_ basePrompt: String) -> String {
    let prefix = referenceEditMode.promptPrefix(
      hasVideo: inputVideoURL != nil,
      hasImages: !inputImageURLs.isEmpty,
      hasAudio: audioInputURL != nil,
      targetDescription: referenceEditTargetDescription
    )
    guard !prefix.isEmpty else { return basePrompt }
    return prefix + "\n\nuser_prompt:\n" + basePrompt
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
    if scope == .faceOnly, !inputImageURLs.isEmpty {
      detectFaces()
    } else if scope == .wholeImage {
      resetFaceReferences()
    }
  }

  func detectFaces() {
    guard !isRunning, !inputImageURLs.isEmpty else { return }
    resetFaceReferences()
    let sourceURLs = inputImageURLs
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

  private func resetVideoMaskCandidates() {
    videoMaskCandidateTask?.cancel()
    videoMaskCandidateTask = nil
    isDetectingVideoMaskCandidates = false
    videoMaskCandidates = []
    selectedVideoMaskCandidateID = nil
    if let videoMaskCandidateDirectory {
      try? FileManager.default.removeItem(at: videoMaskCandidateDirectory)
    }
    videoMaskCandidateDirectory = nil
  }

  private func faceReferencePrompt(_ originalPrompt: String) -> String {
    MiniMaxH3FaceReferenceProcessor.faceOnlyPrompt(
      originalPrompt,
      references: selectedFaceReferences
    )
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

  private static func qwenSequenceLength(in path: String) -> Int? {
    guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
      let object = try? JSONSerialization.jsonObject(with: data),
      let dictionary = object as? [String: Any],
      let qwen = dictionary["qwenComposite"] as? [String: Any],
      let sequenceLength = qwen["sequenceLength"] as? Int,
      sequenceLength > 0
    else { return nil }
    return sequenceLength
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

  private func refreshQwenSequenceLength() {
    qwenSequenceLength = Self.qwenSequenceLength(in: effectiveManifestPath) ?? 4152
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

  func applyModelSetupManifest(_ manifest: String) {
    guard !isRunning,
      !manifest.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    else { return }
    let resolved = Self.resolvePipelineManifestPath(manifest)
    guard Self.isPipelineManifest(URL(fileURLWithPath: resolved)) else {
      status = "MiniMax H3の自動設定manifestを確認できませんでした"
      return
    }
    manifestPath = resolved
    status = "MiniMax H3 manifestを自動設定しました"
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
    let runtimeImageURLs = !inputImageURLs.isEmpty
      && imageReferenceScope == .faceOnly
      ? selectedFaceReferences.map(\.cropURL)
      : inputImageURLs
    let runtimeImageSubjects = !inputImageURLs.isEmpty
      && imageReferenceScope == .faceOnly
      ? selectedFaceReferences.map(\.subjectIndex)
      : []
    let baseRuntimePrompt = !inputImageURLs.isEmpty
      && imageReferenceScope == .faceOnly
      ? faceReferencePrompt(prompt)
      : prompt
    let runtimePrompt = promptWithReferenceEditInstructions(baseRuntimePrompt)
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
    if !runtimeImageURLs.isEmpty {
      do {
        let data = try JSONEncoder().encode(runtimeImageURLs.map(\.path))
        arguments += [
          "--input-images-json",
          String(decoding: data, as: UTF8.self),
        ]
        if !runtimeImageSubjects.isEmpty {
          let subjectsData = try JSONEncoder().encode(runtimeImageSubjects)
          arguments += [
            "--input-image-subjects-json",
            String(decoding: subjectsData, as: UTF8.self),
          ]
        }
      } catch {
        status = "画像入力の準備に失敗しました"
        appendLog("\(error.localizedDescription)\n")
        return
      }
    }
    if let video = inputVideoURL {
      arguments += ["--input", video.path]
    }
    if referenceEditMode != .none {
      arguments += [
        "--reference-edit-mode", referenceEditMode.rawValue,
        "--physical-reference-mask", "1",
      ]
      let target = referenceEditTargetDescription
        .trimmingCharacters(in: .whitespacesAndNewlines)
      if !target.isEmpty {
        arguments += ["--reference-edit-target", target]
      }
      if let selectedVideoMaskCandidateIndex {
        arguments += [
          "--reference-edit-target-index",
          String(selectedVideoMaskCandidateIndex),
        ]
      }
    }
    task.arguments = arguments
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
  let presentModelSetup: () -> Void

  private var faceSwapBinding: Binding<Bool> {
    Binding(
      get: { controller.referenceEditMode == .faceSwap },
      set: { controller.setReferenceEditMode(.faceSwap, enabled: $0) }
    )
  }

  private var bodySwapBinding: Binding<Bool> {
    Binding(
      get: { controller.referenceEditMode == .bodySwap },
      set: { controller.setReferenceEditMode(.bodySwap, enabled: $0) }
    )
  }

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
        if !controller.inputImageURLs.isEmpty {
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
        if controller.inputVideoURL != nil, !controller.inputImageURLs.isEmpty {
          LabeledContent("動画編集") {
            HStack(spacing: 18) {
              Toggle("Face Swap", isOn: faceSwapBinding)
              .toggleStyle(.checkbox)
              Toggle("Body Swap", isOn: bodySwapBinding)
              .toggleStyle(.checkbox)
            }
            .disabled(controller.isRunning)
          }
          VStack(alignment: .leading, spacing: 6) {
            Text("マスク指定")
              .font(.caption)
              .foregroundStyle(.secondary)
            TextField(
              "例: 左の男性、赤い服の人物、中央の人物",
              text: $controller.referenceEditTargetDescription
            )
            .textFieldStyle(.roundedBorder)
            .frame(maxWidth: .infinity)
            .disabled(controller.isRunning || controller.referenceEditMode == .none)
          }
          if controller.referenceEditMode != .none {
            LabeledContent("検出人物") {
              HStack(spacing: 10) {
                if controller.isDetectingVideoMaskCandidates {
                  ProgressView().controlSize(.small)
                  Text("検出中")
                } else if controller.videoMaskCandidates.isEmpty {
                  Text("未検出")
                    .foregroundStyle(.secondary)
                  Button("再検出", action: controller.detectVideoMaskCandidates)
                    .disabled(controller.isRunning)
                } else {
                  Text("\(controller.videoMaskCandidates.count)件")
                    .monospacedDigit()
                  Button("再検出", action: controller.detectVideoMaskCandidates)
                    .disabled(controller.isRunning)
                }
              }
            }
            if !controller.videoMaskCandidates.isEmpty {
              ScrollView(.horizontal) {
                HStack(alignment: .top, spacing: 10) {
                  ForEach(controller.videoMaskCandidates) { candidate in
                    MiniMaxH3VideoMaskCandidateView(
                      controller: controller,
                      candidate: candidate
                    )
                  }
                }
                .padding(.vertical, 2)
              }
            }
          }
          Text(
            "選択すると<Video 1>を元動画、<Picture 1>/<Subject 1>を置換先として扱うRef2VA用プロンプトを実行時に自動追加します。複数人の動画ではマスク指定に置換対象を書いてください。"
          )
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        if controller.inputImageURLs.count > 1 {
          DisclosureGroup("選択した画像（\(controller.inputImageURLs.count)枚）") {
            ForEach(controller.inputImageURLs, id: \.path) { url in
              Text(url.path)
                .font(.caption.monospaced())
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
          }
        }
        if controller.isImageSequence {
          LabeledContent("リップシンク音源") {
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
              "元音源を音声latentとして生成中も固定し、完成動画にも同じ音源をそのまま使用します。1ショットは最大10秒です。"
            )
              .font(.caption)
              .foregroundStyle(.secondary)
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
            Button("モデル自動設定…", action: presentModelSetup)
              .disabled(controller.isRunning)
          }
        }
        if !controller.supportsRuntime {
          Text("Core AI版MiniMax H3にはmacOS 27以降が必要です")
            .foregroundStyle(.red)
        } else if !controller.modelReady {
          Text("モデルは内蔵されません。外部のMiniMax H3 manifest.jsonを選択してください")
            .foregroundStyle(.orange)
        }
      }
      Section("ローカルAIプロンプト生成") {
        LabeledContent("AI") {
          HStack(spacing: 10) {
            Picker("", selection: $controller.promptAssistantProvider) {
              ForEach(MiniMaxH3PromptAssistantProvider.allCases) { provider in
                Text(provider.label).tag(provider)
              }
            }
            .labelsHidden()
            .frame(width: 130)
            TextField("URL", text: $controller.promptAssistantEndpoint)
              .textFieldStyle(.roundedBorder)
            TextField("モデル", text: $controller.promptAssistantModel)
              .textFieldStyle(.roundedBorder)
              .frame(width: 130)
          }
        }
        VStack(alignment: .leading, spacing: 6) {
          Text("日本語の指示")
            .font(.caption)
            .foregroundStyle(.secondary)
          TextEditor(text: $controller.promptAssistantInstruction)
            .font(.body)
            .frame(minHeight: 54, maxHeight: 96)
            .overlay(
              RoundedRectangle(cornerRadius: 6)
                .stroke(Color.secondary.opacity(0.18), lineWidth: 1)
            )
        }
        HStack {
          Button {
            controller.generatePromptWithLocalAI()
          } label: {
            if controller.isGeneratingPrompt {
              ProgressView().controlSize(.small)
              Text("生成中")
            } else {
              Text("H3プロンプトを生成")
            }
          }
          .disabled(
            controller.isGeneratingPrompt || controller.isRunning
              || controller.promptAssistantInstruction
                .trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
          )
          Button("下のプロンプトに反映") {
            controller.applyGeneratedPrompt()
          }
          .disabled(
            controller.isRunning
              || controller.promptAssistantOutput
                .trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
          )
          Spacer()
        }
        VStack(alignment: .leading, spacing: 6) {
          Text("生成されたH3プロンプト")
            .font(.caption)
            .foregroundStyle(.secondary)
          TextEditor(text: $controller.promptAssistantOutput)
            .font(.body)
            .frame(minHeight: 72, maxHeight: 140)
            .overlay(
              RoundedRectangle(cornerRadius: 6)
                .stroke(Color.secondary.opacity(0.18), lineWidth: 1)
            )
        }
        Text(
          "llama.cpp server は通常 http://127.0.0.1:18080/v1/chat/completions を使います。生成結果を確認してから反映してください。"
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
            ? "自由入力（最大\(controller.qwenSequenceLength) Qwenトークン。超過分は末尾を省略）"
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
            : "リップシンク時は1ショット2〜10秒（1080pは6秒固定）です"
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

private struct MiniMaxH3VideoMaskCandidateView: View {
  @ObservedObject var controller: MiniMaxH3Controller
  let candidate: MiniMaxH3VideoMaskCandidate

  private var selected: Bool {
    controller.selectedVideoMaskCandidateID == candidate.id
  }

  var body: some View {
    Button {
      controller.selectVideoMaskCandidate(candidate.id)
    } label: {
      VStack(spacing: 6) {
        if let preview = NSImage(contentsOf: candidate.previewURL) {
          Image(nsImage: preview)
            .resizable()
            .scaledToFill()
            .frame(width: 86, height: 86)
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .overlay(
              RoundedRectangle(cornerRadius: 6)
                .stroke(selected ? Color.accentColor : .secondary.opacity(0.35), lineWidth: selected ? 3 : 1)
            )
        } else {
          RoundedRectangle(cornerRadius: 6)
            .fill(Color.secondary.opacity(0.12))
            .frame(width: 86, height: 86)
            .overlay(Image(systemName: "person.crop.rectangle"))
        }
        HStack(spacing: 4) {
          Image(systemName: selected ? "checkmark.circle.fill" : "circle")
            .foregroundStyle(selected ? Color.accentColor : Color.secondary)
          Text(candidate.label)
        }
        .font(.caption)
        Text("信頼度 \(Int((candidate.confidence * 100).rounded()))%")
          .font(.caption2)
          .foregroundStyle(.secondary)
      }
      .frame(width: 100)
      .contentShape(Rectangle())
    }
    .buttonStyle(.plain)
    .disabled(controller.isRunning)
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
