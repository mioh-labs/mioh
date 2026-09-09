# mioh ローカルクラスタと iPhone / iPad クライアント

## 構成

- **Coordinator Mac**: 動画を時系列shardへ分割し、Workerへ配布して映像片と元音声を最終結合します。
- **Worker Mac**: 選択したCore AI復元・検出モデルでshardを処理します。
- **iPad Worker**: iPadOS 27以降の実機iPadで、同梱した可変長/T18/T36/T90 BasicVSR++と対応する検出モデルを使ってshardを処理します。
- **iPhone / iPadクライアント**: Macの操作、進捗確認、復元プレビューのHLS視聴を行います。iPhoneはWorkerにはなりません。

計算ジョブはBonjourの`_mioh-worker._tcp`、操作・HLS視聴は`_mioh._tcp`として別々に検出されます。操作用認証情報、Capability URL、ローカルのファイルパスはBonjourへ広告しません。Workerは信頼できるLAN内で自動接続し、ペアリングコードは使いません。

## 標準転送: Coordinator HTTP v1

`coordinator-http-v1`が標準です。Coordinator Macがジョブごとに短命なCapability URLを発行し、Workerは次の順序で処理します。

1. AVFoundationのカスタムResource Loaderが64 KiB単位のHTTP Rangeで、担当時間範囲に必要な入力だけを読みます。入力全編の事前ダウンロードは行いません。
2. 完成したvideo-only MP4 shardをWorker内の一時ファイルへ確定します。
3. 正確な`Content-Length`を付けたPUTでCoordinatorへ返します。
4. Coordinatorがチケット、lease、入力元、上限サイズを検証し、成功したshardだけを原子的に公開します。

この方式ではMacとiPadに共通のSMB/NASパスは不要です。入力動画と最終出力へアクセスできるのはCoordinator Macだけで構いません。Capability URLはBearer認証情報と同じ扱いで、ログやUIへ表示しません。

各Range応答は`206 Content-Range`、`Content-Length`、入力SHA-256を表す`ETag`をWorkerが照合します。別URLへのredirectは拒否し、ジョブ取消またはlease失効時は進行中のRange要求も停止します。

## 任意の代替転送: 共有ルート v1

`shared-root-v1`は、CoordinatorとWorkerの両方から同じNAS、SMB共有、または共有ボリュームを直接参照できる場合のフォールバックです。各端末で絶対パスが異なっても構いませんが、設定する「共有ルートID」とルート以下の相対パスは一致させます。

iPadではFilesから共有フォルダを選択します。HTTP転送が使える通常構成では、iPad側の共有ルート設定は不要です。

## Coordinator Mac

1. 「設定」→「ローカル復元クラスタ」で役割を「Coordinator」にし、探索を開始します。
2. 検出されたWorkerから使用する端末を選びます。
3. 入力と出力を選び、「クラスタで書き出す」を有効にして開始します。

HTTP Workerを使うだけなら共有ルートの指定は不要です。共有ルートを設定した場合は、互換Worker向けのフォールバックとしても利用できます。

時系列ジョブは復元バッチのstride境界で分割され、各Workerは先頭にウォームアップ用ハローを読み込みます。ハローは復元状態には使われますが成果物から除外されます。Coordinatorはshardを時系列順に結合し、元動画の音声を最後に一度だけ付加します。

v1では分散境界ごとのサンプリング位相を変えないため、FPS変換との同時使用を拒否します。Worker間で画質を揃えるため、復元モデル・検出モデルとその資産SHA-256が一致するWorkerだけを使用します。

## Worker Mac

1. 役割を「Worker」にします。
2. Workerサービスを開始します。

同じ信頼済みLAN上のCoordinatorがBonjourで自動検出するため、認証コードの登録はありません。

HTTP転送だけなら共有ルートは不要です。SMB/NASを直接読む代替経路も使う場合だけ、共有ルートと共有ルートIDを設定します。Workerは実行可能なモデル資産を起動時に検証し、ジョブ指定のモデルIDと資産SHA-256が一致しなければ処理を拒否します。

## iPad Worker

`apps/MiohRemote/MiohRemote.xcodeproj`をXcode 27で署名し、iPadOS 27以降の**実機iPad**へインストールします。Simulator、iPhone、Designed for iPadとしてMac上で動くアプリは計算Workerになりません。

1. 「Worker」タブで同梱モデルの検証完了を確認します。
2. 「Workerを開始」を押し、処理中はアプリを前景に保ちます。

iPad Workerの対応範囲は次のとおりです。

- 入力: MP4 / MOV / M4V
- 復元: BasicVSR++ 可変長 / T18 / T36 / T90
- 検出: v2 / v3.1-fast / v3.1-accurate / v4-fast / v4-accurate / VR-v2-accurate
- 復元clip長: 1〜90フレーム
- 同時ジョブ: 1件
- 出力: video-only MP4 shard（音声結合はCoordinator Mac）
- 非対応: MKV、ROIエンハンサー、シャープ/ディテール/テクスチャ/スムージング/拡大後処理、FPS変換

Coordinator側では、iPadがBonjour capabilityとして通知する復元・検出モデルと90以下のclip長を選び、上記の非対応効果を無効にしてください。異なる設定のジョブは、画質を黙って変えずに「非互換」として拒否されます。

iPadOSのバックグラウンド実行制限に従い、画面ロックまたはアプリのバックグラウンド移行時には進行中ジョブを停止します。現時点では前景の実機iPadでのみ使用してください。

## iPhone / iPadからの操作とHLS視聴

リモート操作だけならiOS 16以降で利用できます。

1. Mac版miohの設定で「同じLANからmiohを操作する」を一度だけ有効にします。以後はMac起動時に自動で待ち受けます。
2. MacとiPhone/iPadを同じLANへ接続します。
3. mioh Remoteが見つけたMacをタップします。Macが1台で接続情報を保存済みなら、自動で接続します。
4. 初回だけMacに表示された12文字のアクセスコードを入力します。入力が完了すると接続し、次回からKeychainのコードを使用します。

Bonjourで見つからない場合だけ「手動接続」を開き、`http://Macのアドレス:8888`を入力してください。

利用できる機能:

- 再生、一時停止、停止、シーク
- 音量、ミュート
- 書き出し開始・停止、進捗確認
- 音声付きHLS復元プレビュー

入力動画、出力先、復元モデルなどの詳細設定はMac版miohで行います。

## セキュリティと運用上の制約

- **信頼できる家庭・社内のローカルLAN内だけで使用してください。** 制御API、HLS、クラスタHTTP転送にはTLSがありません。ルーターのポート転送、リバースプロキシ、VPN外への公開などを使って、インターネットへ直接公開しないでください。公共Wi-Fiや敵対的な共有ネットワークでも使用しないでください。
- 操作用の12文字アクセスコードはクライアント端末のKeychainへ保存されます。クラスタWorkerはペアリングコードを使わないため、必ず信頼できるLAN内だけで起動してください。
- クラスタHTTP転送の入力と出力は、同一のHTTPオリジン・同一チケット・固定パスでなければWorkerが拒否します。HTTPS、別ホスト/別ポート、userinfo、query、fragment、パス偽装は受け付けません。
- ジョブにはleaseがあり、重複attempt、期限切れ、同一出力の競合、不正なモデル資産、サイズ超過の成果物を拒否します。
- Coordinatorは完成したshardだけを採用します。途中ファイルや失敗したPUTは最終出力へ公開しません。

## ビルド確認

- `packages/MiohRemoteKit`は`swift test`で通信契約、ジョブ台帳、HTTP staging/upload、共有ルートの原子的公開を確認します。
- `apps/MiohRemote/MiohRemote.xcodeproj`の共有Scheme `MiohRemote`を使用します。
- リモート操作クライアントはiOS 16以降、WorkerのビルドはXcode 27 SDK、Worker実行はiPadOS 27以降の実機iPadが必要です。
