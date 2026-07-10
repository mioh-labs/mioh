v1

## User Profile

The user uses Codex as both an execution partner and a durable memory layer across recurring workflows: LADA Apple Silicon and ML/video work, TOORICHO event/news drafting with PDF deliverables, shared-host mail and Apache operations, local Mac operational questions, deploy/sync work, and risk-first code review. They want answers grounded in current live state: Maildir, DB rows, logs, configs, running services, rendered artifacts, disk usage, deployed files, observed data quality, and the actual machine limits of this Mac. They often ask repeated defaults to be remembered when that removes steering, and they correct scope quickly when an answer drifts into the wrong artifact or over-generalizes from stale assumptions. Good collaboration here means moving from diagnosis to execution once asked, keeping edits tightly scoped, verifying on the real target environment, and preserving durable lessons in memory so they do not need to restate the workflow.

## User preferences

- 進捗更新と最終回答は、この環境では日本語をデフォルトにする。ユーザーが別言語を求めた時だけ切り替える。 [ad-hoc note]
- 運用系の質問では、意図した構成ではなく現在の live state から答える。Maildir、DB、log、config、process、disk、rendered artifact を先に確認する。
- `やってください。`, `全部消去して`, `本サーバへは？` のように実行フェーズへ入ったら、診断だけで止めず実作業と検証まで進める。
- `評価して` はサマリ依頼ではなくレビュー依頼として扱い、具体的な壊れ方や運用リスクを優先して返す。
- サーバーで動くローカル編集は、完了条件にアップロード or sync とサーバー側確認まで含める。 [ad-hoc note]
- ネットワーク系の問いでは、`ぽーとじゃなくって、IPアドレス` のような訂正が入りやすいので、ports / IPs / hosts のどれを返すか短く言い直してから答える。
- ローカル LLM の相談は理論最大より快適さ優先。この Mac の実メモリや実運用感に合わせて無理のないサイズを勧める。
- LADA では、設計説明だけで止まらず実装と検証まで一気に進め、CLI 変更は `lada-cli` だけでなく `process_video_parallel.py` まで揃える。
- LADA の dataset / training 系では、抽象説明だけで終えずそのまま実行できるコマンドを先に出す。`本データーのコマンド出して`, `で、コマンドは？` の流れになりやすい。
- TOORICHO の丸亀イベントニュースは draft-only を守り、official source 優先、duplicate check 先行、`eventDateRaw` は実際の開始日時、権利不明画像は使わない。
- 丸亀イベントニュース実行時は、ニュース登録の有無に関係なく Desktop の日本語 PDF を必ず残し、checked sources / skipped candidates / image decisions / warnings / verified PDF path を含める。
- 共有メール作業では、`sa-learn` 成功表示だけで済ませず active Bayes DB と実スコアで反映確認する。sender/domain 証拠を優先し、ブランド名だけで広く学習しない。

## General Tips

- まず [MEMORY.md](/Users/okatti/.codex/memories/MEMORY.md) を cwd 起点で引く。特に `/Users/okatti/Documents/lada`, `/Users/okatti/Documents/tooricho`, `/Users/okatti/Documents/Server`, `/Users/okatti/Documents/gbuc_modern`, `/Users/okatti/Documents/jumbo` が高頻度。
- incremental 更新では `phase2_workspace_diff.md` が最優先。追加 rollout と ad-hoc note を先に飲み込み、削除シグナルだけを外科的に反映する。
- 丸亀イベントニュースは [skills/marugame-event-news-drafting/SKILL.md](/Users/okatti/.codex/memories/skills/marugame-event-news-drafting/SKILL.md) を起点にし、PDF は [skills/japanese-pdf-verification/SKILL.md](/Users/okatti/.codex/memories/skills/japanese-pdf-verification/SKILL.md) で `pdfinfo` + `pdftotext` + `pdftoppm` まで通す。
- `pdftoppm` の `Missing language pack for 'Adobe-Japan1' mapping`, `Unknown font tag`, `No font in show` は失敗扱い。CID フォント PDF は生成できても unreadable なことがある。
- 共有メールの live triage は [skills/server-mail-spam-triage/SKILL.md](/Users/okatti/.codex/memories/skills/server-mail-spam-triage/SKILL.md) が最短。amavis Bayes DB `/var/spool/amavisd/.spamassassin/bayes` に学習し、必要なら `--dump magic` と実サンプル再スコアで反映確認する。 [ad-hoc note]
- この Mac の共有サーバー接続既定は `root@192.168.1.2` + `/Users/okatti/.ssh/gbuc_rsync_ed25519` + `-o IdentitiesOnly=yes`。deploy path や PM2 名は毎回 live state から再確認する。 [ad-hoc note]
- この Mac は `arm64` macOS `26.6` / 16 GB RAM。ローカルモデル相談はその制約に合わせ、LAN 可視化は `route -n get` + `ifconfig` + `arp -a` の順が速い。
- sudden disk loss では `df -h` の後に `du -sh ~/.cache/huggingface` を優先し、`~/llama_models` のような active model root は勝手に消さない。
- LADA 系は `pytest` 前提にせず、まず `python -m unittest` が使えるかを見る。`process_video_parallel.py` の独立 argparse / output semantics と `apply_lada_patches.py` の環境 fix も忘れない。
- LADA の dataset/filter work では、source path と weight path (`lada/model_weights/3rd_party/DOVER.pth`, watermark model, `640m.pt`) の実在確認を先に行う。extractor は candidate generator で、良い学習データの自動保証ではない。

## What's in Memory

### /Users/okatti/Documents/Server and Mac-wide local operations

#### 2026-07-04

- Ornith sizing, ARP host list, and dual-Ethernet dock spot-check: Ornith, ornith:9b, arp -a, 192.168.11.*, en0, OWC Thunderbolt 5 Dual 10GbE Network Dock
  - desc: Search first when the task is a Mac-local model-sizing, quick LAN visibility, or “does this Mac-compatible network accessory exist?” question on this workstation.
  - learnings: this Mac is `arm64` macOS `26.6` with 16 GB RAM, so Ornith 9B is the comfort default; answer exact network artifacts, and dedupe ARP output before replying.

### /Users/okatti/Documents/gbuc_modern

#### 2026-07-04

- Remote M1 track-evaluation migration checklist and live worker inventory: com.gbuc.track-evaluation-webhook, llama-server, TRACK_EVALUATION_WEBHOOK_URL, gbuc-ai-eval-3.12, gbuc_rsync_ed25519, 8788, 18080
  - desc: Search first for `cwd=/Users/okatti/Documents/gbuc_modern` when moving the evaluation worker to a new PC or checking which LaunchAgent, model, env, and key pieces must move together.
  - learnings: inspect the live worker before changing anything, preserve the `8788` webhook and `18080` llama-server contract, and do not retire the old worker until the new machine passes a real test track.

### /Users/okatti/Documents/lada

#### 2026-07-01

- LADA Apple/CoreML follow-up, restore smoothing, output-path fix, and Real-ESRGAN install flow: CoreML, mps-deform-conv, restore-smooth-strength, resolve_single_output_path, apply_lada_patches.py, restore-roi-enhancer-tile
  - desc: Search first for the freshest LADA Apple/MPS work in `cwd=/Users/okatti/Documents/lada` when CoreML detection, `process_video_parallel.py` plumbing, single-file `--output` ffmpeg failures, or ROI enhancer dependency/tuning questions come up together.
  - learnings: the CoreML model is detect-only so downstream code needs box-to-mask compatibility, `process_video_parallel.py` is its own interface, and the practical ROI enhancer starting point is `scale 2 / strength 0.20-0.25 / tile 128`.

### /Users/okatti/Documents/Server

#### 2026-07-03

- `api.cafeyu.xyz` certbot renewal diagnosis and monthly timer override: certbot renew, api.cafeyu.xyz, httpd2.service, Connection refused, certbot-renew.timer, OnCalendar=monthly
  - desc: Search first in `cwd=/Users/okatti/Documents/Server` when Let's Encrypt renewals fail on this host, when Apache listener state may be the blocker, or when the user asks to make cert renewal automatic without noisy default cadence.
  - learnings: check live listeners and certbot logs before touching webroot config; this host uses `httpd2.service` plus `certbot-renew.timer`, and the validated low-noise path was a monthly systemd drop-in with pre/deploy hooks.

### Older Memory Topics

#### /Users/okatti/Documents/lada

- LADA mosaic restoration data prep, filtering, fine-tuning, and processed-file cleanup: create-mosaic-restoration-dataset.py, mps_deform_conv, DOVER, done_processing.txt, /Volumes/Firewire_HD3/movies/FC2/
  - desc: Use for current LADA dataset/training work in `cwd=/Users/okatti/Documents/lada`, especially FC2 extraction, filter-path debugging, BasicVSR++ fine-tuning on MPS, threshold justification, ASCII-safe source cleanup, or moving only fully processed source videos.

- LADA VR video viewer brainstorming entrypoints: VR video viewer, lada/gui/watch, watch_view.py, timeline.py, gstreamer_pipeline_manager.py
  - desc: Use when the user wants a viewer or VR-viewer feature in `cwd=/Users/okatti/Documents/lada` and you need the existing watch/UI entrypoints plus the reminder that the prior rollout was routing-only, not architecture approval.

- LADA MLX ROI seam fix, progress logging, and tuning boundaries: MLX, ROI splitting, split_bbox_by_max_area, process_video_parallel.py, roi=, area=max/sum, window=20
  - desc: Use for MLX restore-path work in `cwd=/Users/okatti/Documents/lada`, especially visible ROI seams, parent `[MLX]` progress output, memory-pressure follow-up, and conceptual streaming/compression questions.

- LADA CoreML backend and masked ROI enhancement: --mosaic-detection-backend coreml, --restore-effect-upscale, empty_lookahead, riddhimanrana/yolo11n-coreml, mps-deform-conv
  - desc: Use for Apple-Silicon LADA work in `cwd=/Users/okatti/Documents/lada` when you need the detect-only CoreML compatibility path, `mps-deform-conv` routing, or masked high-resolution ROI restore behavior that still preserves empty-lookahead skipping.

#### /Users/okatti/Documents/Server and Mac-wide local operations

- Mac-wide operational defaults and production sync completion: launchctl, /opt/homebrew/bin/rsync, root@192.168.1.2, gbuc_rsync_ed25519, server-side verification
  - desc: Use when a task starts on this Mac and needs shared-host transport defaults, LaunchAgent triage, bounded evidence-gathering habits, PDF-verification defaults, or the rule that server-run local edits are not done until synced and checked server-side. [ad-hoc note]

- Mac disk cleanup and model-cache forensics: huggingface cache, GemmaMenuChat, pixtral-12b-4bit, gemma-4-12B, df -h, du -sh, llama_models
  - desc: Use for sudden local disk loss on this Mac when a model download seems to consume tens of gigabytes and confirmed cache paths must be separated from protected active model directories.

#### /Users/okatti/Documents/Server

- Apache topmode-okada.jp vhost addition: topmode-okada.jp, httpd-vhosts.conf, php8-fpm, httpd2.service, curl -Ik, typoでした
  - desc: Use for live Apache vhost additions in `cwd=/Users/okatti/Documents/Server`, especially when HTTP and HTTPS need to be added together with existing Let’s Encrypt certs and the exact `ServerName` matters.

- Shared-mail maintenance, spam hardening, and Spark sync optimization: gbuc_spam_watch.py, doveadm move, D_DISCARD, QUARANTINE_COUNT 0, Spark
  - desc: Use for recurring shared-mail maintenance outside the latest Bayes/rule-tuning run across `cwd=/Users/okatti/Documents/Server` and related shared-mail operations.

#### /Users/okatti/Documents/booked and /Users/okatti/Documents/booked_api

- booked local LLM setup, dashboard changes, commits, and deploy: ollama, LM Studio, admin-dashboard.html, DELETE /admin/bookings/:bookingId, cafeyu_api
  - desc: Use for mixed `booked` / `booked_api` work spanning local Gemma setup, dashboard behavior changes, multi-repo commit hygiene, and production deploy verification.

#### /Users/okatti/Documents/jumbo

- kakaku_api risk-first code review: kakaku_api, ApiClient.php, config.php, item_update.php, realtime_update.php, JSON decode, XML escaping
  - desc: Use when the user asks to evaluate `kakaku_api` and you need the concrete sync-path risks, high-risk files, and the search pattern that avoids minified UI noise in `cwd=/Users/okatti/Documents/jumbo`.

#### /Users/okatti/Documents/tooricho

- Marugame event-news draft registration, PDF font fallback, and broader automation rules: TOORICHO, eventDateRaw, AppleGothic.ttf, draft-only, official sources, duplicate check
  - desc: Use for the broader TOORICHO Marugame workflow in `cwd=/Users/okatti/Documents/tooricho`, including draft registration, duplicate suppression, rights-safe image handling, Desktop PDF verification, and automation-memory writeback.

#### /Users/okatti/Documents/tooricho_api

- TOORICHO AI draft required-note and OCR enforcement: REQUIRED_DRAFT_NOTE, ai下書きの指示note, services/aiContentService.js, services/materialOcrService.js, Tesseract
  - desc: Use for backend AI draft-generation changes in `cwd=/Users/okatti/Documents/tooricho_api` when the task involves required draft-note enforcement, OCR-assisted flyer reading, or PM2 deployment verification.

#### /Users/okatti/Documents/商店街の問題

- Daily central-city revitalization monitoring contract and PDF export: 中心市街地活性化, automation-3, ニュース監視, 更新理由, 新規性の高い情報は限定的, /Users/okatti/Desktop
  - desc: Use when the task is the recurring daily monitor in `cwd=/Users/okatti/Documents/商店街の問題` and must stay separate from the 6am deep dive while producing only new or updated items plus a Desktop PDF.
