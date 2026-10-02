# 動画読み込み方式と診断ログ

「分割」タブの「読み込み・診断」で、次回のローカル書き出し・プレビューの方式を選択する。
macOS 27 以降の Swift ネイティブ処理に適用され、macOS 26 の MPS 処理には適用されない。
復元モデル、検出器、ROI の処理は変更しない。実行途中での切り替えや、失敗時の自動切り替えは行わない。
クラスタワーカーの読み込み方式は、このローカル設定では変更しない。

| 選択肢 | フレームの読み込み | 用途 |
| --- | --- | --- |
| AVFoundation 非同期（既定） | `AVAssetReaderOutput.Provider.next()` | 従来の動作。設定がない場合もこれを使う |
| AVFoundation 同期（比較用） | `copyNextSampleBuffer()` | 同じ Apple デコーダーで、取得 API の違いを比較する。ソフトウェアデコードを強制するものではない |
| FFmpeg ソフトウェア（比較用） | 同梱 FFmpeg、`-hwaccel none` | Apple のフレームデコード経路と比較する。CPU 負荷や色変換結果が異なる場合がある |

FFmpeg 方式も、トラックの寸法・長さなどの情報取得には AVFoundation を使用する。
そのため AVFoundation がファイル自体を開けないケースまで回避できるものではない。ローカルファイルのみが対象。
各フレームの整数 PTS と time base を受け取り、可変 fps を固定 fps で推測し直さない。
読み込みエラー、途中の寸法変更、不正な時刻、FFmpeg の異常終了はエラーにする。失敗フレームを黙って飛ばさない。

## 診断の使い方

同じ動画・同じ復元設定で「詳細な診断ログ」を有効にし、まず既定方式を試す。
失敗した場合は同期方式、FFmpeg 方式で比較する。エラー直前の数回分とエラー行を保存する。
別方式で成功しても、それだけで「入力が正常」「OS のバグ」と断定はできない。

診断は開始時・入力情報取得時・5 秒間隔・終了時などに JSON で記録する。
処理スレッドが待っている間も、別のタイマーから記録する。診断オフではタイマーを作らない。

- `decoder_backend`：選択した方式。
- `frame_counts` / `last_pts_ns`：デコード・検出・エンコードなどの累計数と直近の動画内時刻。
- `interval_fps`：直前の診断からの各工程のフレーム数増分 ÷ 経過時間。既存の進捗表示の累計平均 fps とは異なる。
- `stage_seconds_total` / `stage_seconds_delta`：完了した工程の累計時間／直前からの増分。
- `active_stage_seconds`：進行中の工程が開始してからの時間。同じ工程が並列実行されていれば最長のもの。
- `stage_max_seconds`：完了した工程で最も長かった時間。
- `ring_buffered_frames` / `ring_capacity`：読み込み済みフレームの待ち数／上限。
- `memory`：ネイティブプロセスのメモリ・累計 CPU 時間、システム全体のスワップ量など。
- `thermal_state`：Apple の温度負荷状態（0＝通常、1＝軽度、2＝深刻、3＝危機的）。
- `codec_subtype`：映像コーデックの数値識別子。素材名や任意のメタデータではない。

工程の読み方：`decode` はフレーム取得、`detect` は検出、`restore_batch` は復元バッチ全体、
`prepare`・`restore_and_enhance`・`compose` は復元内部の準備・復元／補正・合成、
`encode` はエンコーダーへの受け渡し等、`finalize` は最終出力の確定。
`ring_empty_wait` はデコード待ち、`ring_full_wait` は後段待ち、`restore_capacity_wait` は復元枠の空き待ち。
並列実行・内包する工程・待ち時間があるため、これらを合算して単純な処理時間の割合にはできない。

メモリ・CPU 時間は FFmpeg や復元用の子プロセスを含まない。システムのスワップ量は他アプリの影響も受ける。
`stop` は診断の終了であり、書き出し成功を意味しない。成功は従来の書き出し完了イベントで確認する。

新しい診断 JSON には素材名・元動画パス・任意のエラー説明・FFmpeg の生ログを含めない。
診断を有効にした書き出し開始ログでも入出力パスを隠す。ただし、既存ログ全体の匿名化を保証する機能ではないため、共有前には全体を確認する。

## 設定ファイル

`decoderBackend` は `avfoundationAsync` / `avfoundationLegacy` / `ffmpegSoftware`。
`detailedDiagnostics` は真偽値。両方とも省略可能で、従来の非同期方式・診断オフになる。

## エラーコードの注意

`AVFoundationErrorDomain / -11821` は `AVErrorDecodeFailed`。
VideoToolbox の `kVTVideoDecoderBadDataErr` は **-12909** であり、**-12137 ではない**。
`-12137` をその定数として解釈し、動画破損が確定したと扱わない。
定義は Xcode SDK の `AVFoundation.framework/Headers/AVError.h` と
`VideoToolbox.framework/Headers/VTErrors.h` で確認できる。
