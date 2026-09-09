# Local Gemma 4でMiniMax H3プロンプトを書く

`scripts/prompting/write-minimax-h3-with-gemma.py` は、Codex用の
`h3-prompt-writing/SKILL.md` と必要な参照ガイドを読み、ローカルの
Gemma 4へシステムプロンプトとして渡します。Gemma側に専用のSkill機構は不要です。

既定では、既存の `llama-server` のOpenAI互換API
`http://127.0.0.1:18080/v1` と次のスキルを使用します。

```text
/Users/okatti/.codex/skills/h3-prompt-writing
```

## テキストから生成（T2VA）

```bash
cd /Users/okatti/Documents/lada
python scripts/prompting/write-minimax-h3-with-gemma.py \
  --mode t2va \
  --duration 10 \
  --request "夏の夕暮れの海岸を歩く成人女性。実写映画風" \
  --output /Users/okatti/Desktop/h3-prompt.txt
```

## 参照画像から生成（Ref2VA）

`--image` の指定順が `<Picture 1>`, `<Picture 2>` の順になります。

```bash
cd /Users/okatti/Documents/lada
python scripts/prompting/write-minimax-h3-with-gemma.py \
  --mode ref2va \
  --duration 10 \
  --image /absolute/path/person.jpg \
  --image /absolute/path/location.jpg \
  --request "Picture 1の成人人物を維持し、Picture 2の海岸を歩かせる" \
  --output /Users/okatti/Desktop/h3-ref2va-prompt.txt
```

`--mode` は `auto`, `t2va`, `i2va`, `fl2va`, `l2va`, `ref2va` に対応します。
スキルを別の場所へ移した場合は `--skill-dir` または環境変数
`H3_PROMPT_SKILL_DIR` で指定できます。APIは `--api-url` または
`GEMMA_API_URL` で変更できます。

Gemmaへ送る実際のシステムプロンプトだけを確認する場合は
`--print-system-prompt` を付けます。この場合、API呼び出しは行いません。
