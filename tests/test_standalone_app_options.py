import unittest
import plistlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = ROOT / "packaging" / "macOS" / "standalone" / "MiohApp.swift"
PLAYER_SOURCE = ROOT / "packaging" / "macOS" / "standalone" / "RealtimePlayer.swift"
BUILD_SCRIPT = ROOT / "packaging" / "macOS" / "standalone" / "build_app.sh"
INFO_PLIST = ROOT / "packaging" / "macOS" / "standalone" / "Info.plist"


class StandaloneAppOptionTests(unittest.TestCase):
    def test_native_realtime_player_has_buffered_audio_synced_controls(self):
        self.assertTrue(PLAYER_SOURCE.is_file())
        player = PLAYER_SOURCE.read_text()
        app = APP_SOURCE.read_text()

        expected_player_contracts = [
            "import AVFoundation",
            "import AVKit",
            "enum RealtimePlayerState",
            "struct PreviewWorkerEvent: Decodable",
            "final class RealtimePlayerController: ObservableObject",
            "let sourcePlayer = AVPlayer()",
            "let restoredPlayer = AVQueuePlayer()",
            "event.generation == generation",
            "driftToleranceSeconds = 0.080",
            'sendCommand(["command": "seek"',
            "showOriginal",
            "struct RealtimePlayerView: View",
            'Label("再生"',
            'Label("一時停止"',
            'Text("バッファ中")',
        ]
        for contract in expected_player_contracts:
            self.assertIn(contract, player)
        self.assertIn('RealtimePlayerView(controller: player, runner: runner)', app)
        self.assertIn('.tabItem { Label("再生", systemImage: "play.rectangle") }', app)

    def test_player_starts_each_generation_once_and_resumes_without_seeking(self):
        player = PLAYER_SOURCE.read_text()

        for contract in [
            "private var generationHasStarted = false",
            "private var generationStartPending = false",
            "guard state != .playing, !generationStartPending else { return }",
            "private func startPlayersFromCurrentPosition()",
            "let startingGeneration = generation",
            "self.generation == startingGeneration",
        ]:
            self.assertIn(contract, player)

    def test_player_rechecks_latest_buffer_policy_after_initial_seek(self):
        player = PLAYER_SOURCE.read_text()
        resume = player.split("private func resumeIfBuffered", 1)[1].split(
            "private func startPlayersFromCurrentPosition", 1
        )[0]

        latest_check = "let latestPolicyAllowsPlayback = PreviewBufferPolicy.canStart("
        self.assertIn(latest_check, resume)
        for contract in [
            "bufferedSeconds: self.bufferedSeconds",
            "selectedBufferLimit: runner.previewBufferLimit",
            "generationHasStarted: self.generationHasStarted",
            "shortenRebuffer: runner.previewShortenedRebuffer",
            "endOfFile: self.workerGenerationEnded",
            "hasQueuedSegments: !self.queuedSegments.isEmpty",
            "guard latestPolicyAllowsPlayback else",
        ]:
            self.assertIn(contract, resume)
        self.assertLess(
            resume.index(latest_check),
            resume.index("self.generationHasStarted = true"),
        )

    def test_player_retains_generation_eof_until_the_queue_drains(self):
        player = PLAYER_SOURCE.read_text()
        start = player.split("func start(runner:", 1)[1].split(
            "func togglePlayback", 1
        )[0]
        seek = player.split("func seek(to seconds:", 1)[1].split(
            "func restartWithCurrentSettings", 1
        )[0]
        stop = player.split("func stop()", 1)[1].split(
            "private func consumeWorkerOutput", 1
        )[0]
        ended_event = player.split('case "ended":', 1)[1].split(
            'case "error":', 1
        )[0]
        finished = player.split("private func finished", 1)[1].split(
            "private func resumeIfBuffered", 1
        )[0]
        resume = player.split("private func resumeIfBuffered", 1)[1].split(
            "private func startPlayersFromCurrentPosition", 1
        )[0]

        self.assertIn("private var workerGenerationEnded = false", player)
        for reset_scope in [start, seek, stop]:
            self.assertIn("workerGenerationEnded = false", reset_scope)
        self.assertIn("workerGenerationEnded = true", ended_event)
        self.assertIn("if queuedSegments.isEmpty", ended_event)
        self.assertIn("if queuedSegments.isEmpty {", finished)
        self.assertNotIn(
            "if queuedSegments.isEmpty && state == .playing",
            finished,
        )
        self.assertIn("if workerGenerationEnded", finished)
        self.assertIn("shouldPlay = false", finished)
        self.assertIn("state = .ended", finished)
        self.assertIn("endOfFile: workerGenerationEnded", resume)

    def test_player_rejects_callbacks_from_old_worker_sessions(self):
        player = PLAYER_SOURCE.read_text()
        start = player.split("func start(runner:", 1)[1].split(
            "func togglePlayback", 1
        )[0]
        stop = player.split("func stop()", 1)[1].split(
            "private func consumeWorkerOutput", 1
        )[0]
        stdout_callback = start.split(
            "outputPipe.fileHandleForReading.readabilityHandler", 1
        )[1].split("errorPipe.fileHandleForReading.readabilityHandler", 1)[0]
        stderr_callback = start.split(
            "errorPipe.fileHandleForReading.readabilityHandler", 1
        )[1].split("process.terminationHandler", 1)[0]
        termination_callback = start.split("process.terminationHandler", 1)[1]

        self.assertIn("private var activeWorkerSessionToken: UUID?", player)
        self.assertIn("let sessionToken = UUID()", start)
        self.assertIn("activeWorkerSessionToken = sessionToken", start)
        self.assertIn('stdoutBuffer = ""', start)
        for callback in [stdout_callback, stderr_callback, termination_callback]:
            self.assertIn(
                "self.activeWorkerSessionToken == sessionToken",
                callback,
            )
        self.assertIn("self.worker === completed", termination_callback)
        self.assertIn("activeWorkerSessionToken = nil", stop)
        self.assertIn('stdoutBuffer = ""', stop)

    def test_player_tracks_completed_source_seek_per_generation(self):
        player = PLAYER_SOURCE.read_text()
        start = player.split("func start(runner:", 1)[1].split(
            "func togglePlayback", 1
        )[0]
        seek = player.split("func seek(to seconds:", 1)[1].split(
            "func restartWithCurrentSettings", 1
        )[0]
        stop = player.split("func stop()", 1)[1].split(
            "private func consumeWorkerOutput", 1
        )[0]
        resume = player.split("private func resumeIfBuffered", 1)[1].split(
            "private func startPlayersFromCurrentPosition", 1
        )[0]

        self.assertIn("private var generationSourceSeekCompleted = false", player)
        for reset_scope in [start, seek, stop]:
            self.assertIn("generationSourceSeekCompleted = false", reset_scope)
        direct_start = "if generationSourceSeekCompleted {"
        self.assertIn(direct_start, resume)
        self.assertLess(resume.index(direct_start), resume.index("sourcePlayer.seek("))
        direct_start_scope = resume.split(direct_start, 1)[1].split(
            "sourcePlayer.seek(", 1
        )[0]
        self.assertIn("generationHasStarted = true", direct_start_scope)
        self.assertIn("startPlayersFromCurrentPosition()", direct_start_scope)

    def test_source_seek_completions_are_current_finished_and_retryable(self):
        player = PLAYER_SOURCE.read_text()
        seek = player.split("func seek(to seconds:", 1)[1].split(
            "func restartWithCurrentSettings", 1
        )[0]
        resume = player.split("private func resumeIfBuffered", 1)[1].split(
            "private func startPlayersFromCurrentPosition", 1
        )[0]
        initial_completion = resume.split("sourcePlayer.seek(", 1)[1]

        for contract in [
            "guard let seekSessionToken = activeWorkerSessionToken else { return }",
            "let seekGeneration = generation",
            "generationStartPending = true",
            ") { [weak self] finished in",
            "self.activeWorkerSessionToken == seekSessionToken",
            "self.generation == seekGeneration",
            "self.generationStartPending = false",
            "guard finished else {",
            "self.generationSourceSeekCompleted = true",
            "self.resumeIfBuffered()",
        ]:
            self.assertIn(contract, seek)
        self.assertLess(
            seek.index("self.generationStartPending = false"),
            seek.index("guard finished else {"),
        )

        self.assertIn("let startingSessionToken = activeWorkerSessionToken", resume)
        for contract in [
            ") { [weak self] finished in",
            "self.activeWorkerSessionToken == startingSessionToken",
            "self.generation == startingGeneration",
            "self.generationStartPending = false",
            "guard finished else {",
            "self.generationSourceSeekCompleted = true",
        ]:
            self.assertIn(contract, initial_completion)
        self.assertLess(
            initial_completion.index("self.generationStartPending = false"),
            initial_completion.index("guard finished else {"),
        )
        self.assertLess(
            initial_completion.index("guard finished else {"),
            initial_completion.index("self.generationSourceSeekCompleted = true"),
        )

    def test_interrupted_source_seeks_buffer_and_retry_immediately(self):
        player = PLAYER_SOURCE.read_text()
        manual_completion = player.split("func seek(to seconds:", 1)[1].split(
            "func restartWithCurrentSettings", 1
        )[0].split(") { [weak self] finished in", 1)[1]
        buffered_start_completion = player.split(
            "private func resumeIfBuffered", 1
        )[1].split("private func startPlayersFromCurrentPosition", 1)[0].split(
            ") { [weak self] finished in", 1
        )[1]
        retry_branch = "guard finished else {"

        for completion in [manual_completion, buffered_start_completion]:
            self.assertIn(retry_branch, completion)
            self.assertLess(
                completion.index("self.generationStartPending = false"),
                completion.index(retry_branch),
            )
            failure_scope = completion.split(retry_branch, 1)[1].split(
                "self.generationSourceSeekRetryCount = 0", 1
            )[0]
            self.assertIn("self.state = .buffering", failure_scope)
            self.assertIn("self.resumeIfBuffered()", failure_scope)
            self.assertNotIn("generationSourceSeekCompleted", failure_scope)

    def test_source_seek_completions_preserve_terminal_states(self):
        player = PLAYER_SOURCE.read_text()
        manual_completion = player.split("func seek(to seconds:", 1)[1].split(
            "func restartWithCurrentSettings", 1
        )[0].split(") { [weak self] finished in", 1)[1]
        buffered_start_completion = player.split(
            "private func resumeIfBuffered", 1
        )[1].split("private func startPlayersFromCurrentPosition", 1)[0].split(
            ") { [weak self] finished in", 1
        )[1]
        resume = player.split("private func resumeIfBuffered", 1)[1].split(
            "private func startPlayersFromCurrentPosition", 1
        )[0]
        fail = player.split("private func fail", 1)[1].split(
            "struct RealtimePlayerView", 1
        )[0]
        terminal_guard = (
            "guard self.shouldPlay, self.state != .failed, self.state != .ended "
            "else { return }"
        )

        for completion in [manual_completion, buffered_start_completion]:
            self.assertIn(terminal_guard, completion)
            self.assertLess(
                completion.index("self.generationStartPending = false"),
                completion.index(terminal_guard),
            )
            self.assertLess(
                completion.index(terminal_guard),
                completion.index("guard finished else {"),
            )
        self.assertIn("shouldPlay = false", fail)
        self.assertLess(fail.index("shouldPlay = false"), fail.index("state = .failed"))
        self.assertIn(
            "guard state != .failed, state != .ended else { return }",
            resume,
        )

    def test_source_seek_retries_are_bounded_per_generation(self):
        player = PLAYER_SOURCE.read_text()
        start = player.split("func start(runner:", 1)[1].split(
            "func togglePlayback", 1
        )[0]
        seek = player.split("func seek(to seconds:", 1)[1].split(
            "func restartWithCurrentSettings", 1
        )[0]
        stop = player.split("func stop()", 1)[1].split(
            "private func consumeWorkerOutput", 1
        )[0]
        manual_completion = seek.split(") { [weak self] finished in", 1)[1]
        buffered_start_completion = player.split(
            "private func resumeIfBuffered", 1
        )[1].split("private func startPlayersFromCurrentPosition", 1)[0].split(
            ") { [weak self] finished in", 1
        )[1]
        exhausted_message = "プレビューのシークに繰り返し失敗しました"

        self.assertIn("private let maximumGenerationSourceSeekRetries = 2", player)
        self.assertIn("private var generationSourceSeekRetryCount = 0", player)
        for reset_scope in [start, seek, stop]:
            self.assertIn("generationSourceSeekRetryCount = 0", reset_scope)
        for completion in [manual_completion, buffered_start_completion]:
            for contract in [
                "self.generationSourceSeekRetryCount < "
                "self.maximumGenerationSourceSeekRetries",
                "self.generationSourceSeekRetryCount += 1",
                f'self.fail("{exhausted_message}")',
                "self.generationSourceSeekRetryCount = 0",
            ]:
                self.assertIn(contract, completion)
            self.assertLess(
                completion.index("self.generationSourceSeekRetryCount += 1"),
                completion.index("self.state = .buffering"),
            )
            self.assertLess(
                completion.index("guard finished else {"),
                completion.index("self.generationSourceSeekRetryCount = 0"),
            )
            success_scope = completion.split("guard finished else {", 1)[1].split(
                "self.generationSourceSeekCompleted = true", 1
            )[0]
            self.assertEqual(success_scope.count("generationSourceSeekRetryCount = 0"), 1)

    def test_long_video_source_seeks_use_nonzero_named_tolerance(self):
        player = PLAYER_SOURCE.read_text()
        manual_seek = player.split("func seek(to seconds:", 1)[1].split(
            "func restartWithCurrentSettings", 1
        )[0].split("sourcePlayer.seek(", 1)[1]
        buffered_start_seek = player.split("private func resumeIfBuffered", 1)[1].split(
            "private func startPlayersFromCurrentPosition", 1
        )[0].split("sourcePlayer.seek(", 1)[1]
        drift_seek = player.split("private func tick", 1)[1].split(
            "private func updateBufferedDuration", 1
        )[0].split("restoredPlayer.seek(", 1)[1]

        self.assertIn("let sourceSeekToleranceSeconds = 0.25", player)
        self.assertIn(
            "CMTime(seconds: sourceSeekToleranceSeconds, preferredTimescale: 600)",
            player,
        )
        for source_seek in [manual_seek, buffered_start_seek]:
            self.assertIn("toleranceBefore: sourceSeekTolerance", source_seek)
            self.assertIn("toleranceAfter: sourceSeekTolerance", source_seek)
        self.assertEqual(player.count("toleranceBefore: sourceSeekTolerance"), 2)
        self.assertEqual(player.count("toleranceAfter: sourceSeekTolerance"), 2)
        self.assertIn("toleranceBefore: .zero", drift_seek)
        self.assertIn("toleranceAfter: .zero", drift_seek)
        self.assertNotIn("sourceSeekTolerance", drift_seek)

    def test_source_seek_watchdog_releases_missing_avplayer_completions(self):
        player = PLAYER_SOURCE.read_text()
        manual_seek = player.split("func seek(to seconds:", 1)[1].split(
            "func restartWithCurrentSettings", 1
        )[0]
        buffered_start_seek = player.split("private func resumeIfBuffered", 1)[1].split(
            "private func startPlayersFromCurrentPosition", 1
        )[0]
        watchdog = player.split("private func resolveStalledSourceSeek", 1)[1].split(
            "private func startPlayersFromCurrentPosition", 1
        )[0]

        for contract in [
            "private var activeSourceSeekRequestToken: UUID?",
            "private let sourceSeekWatchdogNanoseconds: UInt64 = 2_000_000_000",
            "private let sourceSeekWatchdogToleranceSeconds = 1.0",
        ]:
            self.assertIn(contract, player)
        for source_seek in [manual_seek, buffered_start_seek]:
            self.assertIn("activeSourceSeekRequestToken =", source_seek)
            self.assertIn("Task { @MainActor", source_seek)
            self.assertIn("Task.sleep(nanoseconds: seekWatchdogNanoseconds)", source_seek)
            self.assertIn("resolveStalledSourceSeek(", source_seek)
        for contract in [
            "activeSourceSeekRequestToken == requestToken",
            "activeWorkerSessionToken == sessionToken",
            "generation == seekGeneration",
            "generationStartPending",
            "sourcePlayer.currentTime().seconds",
            "abs(currentSeconds - targetSeconds) <= sourceSeekWatchdogToleranceSeconds",
            "generationSourceSeekCompleted = true",
            "resumeIfBuffered()",
        ]:
            self.assertIn(contract, watchdog)

    def test_all_avplayer_callbacks_are_isolated_to_the_worker_session(self):
        player = PLAYER_SOURCE.read_text()
        start = player.split("func start(runner:", 1)[1].split(
            "func togglePlayback", 1
        )[0]
        enqueue = player.split("private func enqueue", 1)[1].split(
            "private func finished", 1
        )[0]
        time_observer = player.split("private func installTimeObserver", 1)[1].split(
            "private func tick", 1
        )[0]

        self.assertIn("installTimeObserver(sessionToken: sessionToken)", start)
        self.assertIn("(sessionToken: UUID)", time_observer)
        self.assertIn(
            "self.activeWorkerSessionToken == sessionToken",
            time_observer,
        )
        for contract in [
            "guard let itemSessionToken = activeWorkerSessionToken else { return }",
            "let itemGeneration = generation",
            "self.activeWorkerSessionToken == itemSessionToken",
            "self.generation == itemGeneration",
        ]:
            self.assertIn(contract, enqueue)

    def test_completed_item_observers_are_removed_immediately(self):
        player = PLAYER_SOURCE.read_text()
        deinit = player.split("deinit", 1)[1].split("func start(runner:", 1)[0]
        enqueue = player.split("private func enqueue", 1)[1].split(
            "private func finished", 1
        )[0]
        finished = player.split("private func finished", 1)[1].split(
            "private func resumeIfBuffered", 1
        )[0]
        clear_queue = player.split("private func clearRestoredQueue", 1)[1].split(
            "private func cleanupSession", 1
        )[0]

        self.assertIn(
            "private var notificationTokens: [ObjectIdentifier: NSObjectProtocol] = [:]",
            player,
        )
        self.assertIn("for token in notificationTokens.values", deinit)
        self.assertIn("notificationTokens[identifier] = token", enqueue)
        for contract in [
            "let identifier = ObjectIdentifier(item)",
            "notificationTokens.removeValue(forKey: identifier)",
            "NotificationCenter.default.removeObserver(token)",
        ]:
            self.assertIn(contract, finished)
        self.assertLess(
            finished.index("notificationTokens.removeValue(forKey: identifier)"),
            finished.index("itemSegments.removeValue(forKey: identifier)"),
        )
        self.assertIn("for token in notificationTokens.values", clear_queue)
        self.assertIn("notificationTokens.removeAll()", clear_queue)

    def test_playback_input_is_independent_from_export_input(self):
        player = PLAYER_SOURCE.read_text()
        app = APP_SOURCE.read_text()

        for contract in [
            "@Published var previewInputURL: URL?",
            "func choosePreviewInput()",
            "guard let input = previewInputURL",
            'PathRow(title: "再生動画"',
            "controller.previewInputURL == nil",
        ]:
            self.assertIn(contract, player)
        self.assertNotIn("guard let input = runner.inputURL", player)
        self.assertIn(
            "func previewArguments(resources: URL, outputDirectory: URL, input: URL)",
            app,
        )
        self.assertIn(
            'var args = ["--input", input.path, "--output-dir", outputDirectory.path]',
            app,
        )

    def test_app_precompiles_coreml_detection_models(self):
        build_script = BUILD_SCRIPT.read_text()

        self.assertIn("xcrun coremlcompiler compile", build_script)
        self.assertIn('"$RESOURCES/models/$compiled_name"', build_script)

    def test_runner_exposes_current_settings_to_preview_worker(self):
        app = APP_SOURCE.read_text()

        self.assertIn(
            "func previewArguments(resources: URL, outputDirectory: URL, input: URL)",
            app,
        )
        for option in [
            "--restoration-model", "--detection-model", "--max-clip-length",
            "--restore-max-frames", "--sharpen-strength", "--detail-boost",
            "--blend-feather", "--texture-mix", "--smooth-strength",
            "--roi-enhancer", "--roi-enhancer-model", "--roi-enhancer-scale",
            "--roi-enhancer-strength", "--roi-enhancer-tile",
            "--effect-upscale", "--buffer-limit",
        ]:
            self.assertIn(option, app)

    def test_preview_buffer_slider_supports_one_minute_and_live_updates(self):
        app = APP_SOURCE.read_text()
        player = PLAYER_SOURCE.read_text()

        for contract in [
            "@Published var previewBufferLimit = 8.0",
            'add(&args, "--buffer-limit", previewBufferLimit)',
        ]:
            self.assertIn(contract, app)
        for contract in [
            "func setBufferLimit(_ seconds: Double)",
            '["command": "set_buffer_limit", "seconds": seconds]',
            "let seconds: Double?",
            'case "buffer_limit":',
            'runner?.appendExternalLog("プレビューバッファ上限を適用: \\(Int(seconds))秒\\n")',
            'Text("バッファ上限")',
            "in: 1...60",
            "step: 1",
            "controller.setBufferLimit(value)",
            'Text("\\(Int(runner.previewBufferLimit))秒")',
        ]:
            self.assertIn(contract, player)

    def test_player_waits_for_selected_buffer_duration_with_optional_fast_rebuffer(self):
        app = APP_SOURCE.read_text()
        player = PLAYER_SOURCE.read_text()

        self.assertIn("@Published var previewShortenedRebuffer = false", app)
        for contract in [
            "PreviewBufferPolicy.canStart(",
            "bufferedSeconds: bufferedSeconds",
            "selectedBufferLimit: runner.previewBufferLimit",
            "generationHasStarted: generationHasStarted",
            "shortenRebuffer: runner.previewShortenedRebuffer",
            "endOfFile: workerGenerationEnded",
            "hasQueuedSegments: !queuedSegments.isEmpty",
            "func bufferPolicyDidChange()",
            '"再バッファを短縮",',
            ".toggleStyle(.checkbox)",
            "controller.bufferPolicyDidChange()",
        ]:
            self.assertIn(contract, player)
        self.assertNotIn("startupSegmentCount = 3", player)
        self.assertNotIn("rebufferSegmentCount = 2", player)

    def test_preview_uses_t36_without_changing_t90_export_default(self):
        source = APP_SOURCE.read_text()

        self.assertIn("var previewRestorationModel: String", source)
        self.assertIn(
            'supportsCoreAI ? "basicvsrpp-v1.2-coreai-t36" : "basicvsrpp-v1.2"',
            source,
        )
        self.assertIn("let previewModel = capabilities.previewRestorationModel", source)
        self.assertIn('add(&args, "--restoration-model", previewModel)', source)
        self.assertIn("switch previewModel", source)
        self.assertIn(
            'supportsCoreAI ? "basicvsrpp-v1.2-coreai-t90" : "basicvsrpp-v1.2"',
            source,
        )

    def test_main_app_targets_macos_26_without_linking_coreai(self):
        source = APP_SOURCE.read_text()
        build_script = BUILD_SCRIPT.read_text()
        with INFO_PLIST.open("rb") as handle:
            info = plistlib.load(handle)

        self.assertEqual(info["LSMinimumSystemVersion"], "26.0")
        self.assertNotIn("import CoreAI", source)
        self.assertEqual(build_script.count("-target arm64-apple-macosx26.0"), 1)
        self.assertEqual(build_script.count("-target arm64-apple-macosx27.0"), 1)
        self.assertEqual(build_script.count("-framework CoreAI"), 1)

    def test_model_choices_follow_coreai_os_availability(self):
        source = APP_SOURCE.read_text()

        self.assertIn("struct PlatformCapabilities", source)
        self.assertIn("operatingSystemVersion.majorVersion >= 27", source)
        self.assertIn(
            'supportsCoreAI ? "basicvsrpp-v1.2-coreai-t90" : "basicvsrpp-v1.2"',
            source,
        )
        self.assertIn(
            'supportsCoreAI ? coreAIRestorationModels + baseRestorationModels : baseRestorationModels',
            source,
        )
        self.assertIn(
            'supportsCoreAI ? baseDetectionModels + ["v4-fast-coreai"] : baseDetectionModels',
            source,
        )
        self.assertIn("normalizeModelSelections()", source)

    def test_coreai_helper_environment_is_only_exported_when_supported(self):
        source = APP_SOURCE.read_text()

        self.assertIn("if capabilities.supportsCoreAI", source)
        self.assertIn('result["LADA_COREAI_SWIFT_RUNNER"]', source)
        self.assertIn(
            "try rejectUnsupportedCoreAIModel(roiEnhancerModel)",
            source,
        )

    def test_parallel_worker_selection_is_forwarded_without_platform_limit(self):
        source = APP_SOURCE.read_text()

        self.assertIn('add(&args, "--parallel-workers", parallelWorkers)', source)
        self.assertNotIn("min(parallelWorkers", source)
        self.assertNotIn("maxParallelWorkers", source)

    def test_product_is_named_mioh(self):
        self.assertTrue(APP_SOURCE.is_file(), "MiohApp.swift must be the app entry source")
        source = APP_SOURCE.read_text()
        build_script = BUILD_SCRIPT.read_text()
        with INFO_PLIST.open("rb") as handle:
            info = plistlib.load(handle)

        self.assertIn('Text("mioh")', source)
        self.assertIn('PathSettingRow(title: "mioh一時フォルダ"', source)
        self.assertIn("struct MiohStandaloneApp: App", source)
        self.assertEqual(info["CFBundleDisplayName"], "mioh")
        self.assertEqual(info["CFBundleName"], "mioh")
        self.assertEqual(info["CFBundleExecutable"], "mioh")
        self.assertEqual(info["CFBundleIdentifier"], "com.okatti.lada.coreai")
        self.assertIn('APP="$BUILD_DIR/mioh.app"', build_script)
        self.assertIn('-o "$CONTENTS/MacOS/mioh"', build_script)
        self.assertIn('DMG="$BUILD_DIR/mioh-0.11.0-unsigned.dmg"', build_script)
        self.assertIn('--volumeName "mioh"', build_script)
        self.assertIn('ditto "$APP" "$DMG_ROOT/mioh.app"', build_script)

    def test_gui_exposes_all_processing_options(self):
        source = APP_SOURCE.read_text()
        expected_options = {
            "--input", "--output", "--temp-dir", "--ffmpeg-temp-dir",
            "--lada-temp-dir", "--parallel-workers", "--executor",
            "--segment-duration", "--segment-count", "--merge-encoder",
            "--delete-segments", "--keep-temp", "--force-split", "--device",
            "--fp16", "--no-fp16", "--encoding-preset", "--encoder",
            "--encoder-options", "--bitrate-multiplier", "--quality", "--qmin",
            "--qmax", "--fps", "--pre-fps-conversion", "--mp4-fast-start",
            "--auto-optimize", "--no-auto-optimize", "--mosaic-restoration-model",
            "--max-clip-length", "--restore-max-frames",
            "--restore-sharpen-strength", "--restore-detail-boost",
            "--restore-blend-feather", "--restore-texture-mix",
            "--restore-smooth-strength", "--restore-effect-upscale",
            "--restore-roi-enhancer", "--restore-roi-enhancer-model-path",
            "--restore-roi-enhancer-scale", "--restore-roi-enhancer-strength",
            "--restore-roi-enhancer-tile", "--mosaic-detection-model",
            "--mosaic-detection-empty-lookahead", "--detect-face-mosaics",
            "--no-detect-face-mosaics", "--memory-cleanup-interval",
            "--cleanup-trigger-gb", "--mps-memory-fraction", "--log-mps-memory",
            "--overwrite",
        }

        missing = sorted(option for option in expected_options if option not in source)
        self.assertEqual(missing, [])

    def test_app_bundles_parallel_processor(self):
        script = BUILD_SCRIPT.read_text()
        self.assertIn("process_video_parallel.py", script)

    def test_app_bundles_realtime_player_and_preview_worker(self):
        script = BUILD_SCRIPT.read_text()

        self.assertIn('"$PACKAGE_DIR/PreviewBufferPolicy.swift"', script)
        self.assertIn('"$PACKAGE_DIR/RealtimePlayer.swift"', script)
        self.assertIn("-framework AVFoundation", script)
        self.assertIn("-framework AVKit", script)
        self.assertIn(
            '"$RESOURCES/runtime/lib/python3.12/site-packages/mioh_preview_worker.py"',
            script,
        )

    def test_standalone_app_uses_mioh_icon(self):
        icon = ROOT / "lada" / "gui" / "icons" / "mioh-icon.png"
        script = BUILD_SCRIPT.read_text()

        self.assertTrue(icon.is_file())
        self.assertEqual(icon.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertIn('SOURCE_ICON="$ROOT/lada/gui/icons/mioh-icon.png"', script)

    def test_app_replaces_structured_progress_rows(self):
        source = APP_SOURCE.read_text()

        self.assertIn('result["LADA_APP_PROGRESS"] = "1"', source)
        self.assertIn("struct AppProgressEvent: Decodable", source)
        self.assertIn('"@@LADA_PROGRESS@@"', source)
        self.assertIn("activeProgress", source)
        self.assertIn("logHistory", source)
        self.assertIn("rebuildVisibleLog()", source)

    def test_gui_has_always_visible_multiline_ffmpeg_options(self):
        source = APP_SOURCE.read_text()

        self.assertIn('Section("FFmpeg詳細設定")', source)
        self.assertIn('Text("追加FFmpegオプション")', source)
        self.assertIn('TextEditor(text: $runner.encoderOptions)', source)
        self.assertEqual(
            source.count(
                'addOptional(&args, "--encoder-options", encoderOptions)'
            ),
            1,
        )

    def test_restoration_effects_use_slider_number_rows(self):
        source = APP_SOURCE.read_text()
        expected_slider_rows = [
            'doubleSliderField("シャープ", value: $runner.sharpenStrength, range: 0...1, step: 0.05)',
            'doubleSliderField("ディテール", value: $runner.detailBoost, range: 0...1, step: 0.05)',
            'doubleSliderField("境界フェザー", value: $runner.blendFeather, range: 0...3, step: 0.05)',
            'doubleSliderField("テクスチャ", value: $runner.textureMix, range: 0...1, step: 0.01)',
            'doubleSliderField("スムージング", value: $runner.smoothStrength, range: 0...1, step: 0.05)',
            'doubleSliderField("強度", value: $runner.roiEnhancerStrength, range: 0...1, step: 0.05)',
            'integerSliderField("タイル", value: $runner.roiEnhancerTile, range: 0...1024, step: 32)',
        ]

        self.assertIn("private func doubleSliderField", source)
        self.assertIn("private func integerSliderField", source)
        for row in expected_slider_rows:
            self.assertIn(row, source)
        self.assertIn(
            'LabeledContent("エフェクト倍率") { Stepper(value: $runner.effectUpscale, in: 1...4)',
            source,
        )
        self.assertIn(
            'LabeledContent("倍率") { Stepper(value: $runner.roiEnhancerScale, in: 1...8)',
            source,
        )
        self.assertIn(
            'doubleSliderField("強度", value: $runner.roiEnhancerStrength, range: 0...1, step: 0.05).disabled(runner.roiEnhancer == "none")',
            source,
        )
        self.assertIn(
            'integerSliderField("タイル", value: $runner.roiEnhancerTile, range: 0...1024, step: 32).disabled(runner.roiEnhancer == "none")',
            source,
        )


if __name__ == "__main__":
    unittest.main()
