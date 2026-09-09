# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_API_URL = "http://127.0.0.1:18080/v1"
DEFAULT_SKILL_DIR = Path.home() / ".codex" / "skills" / "h3-prompt-writing"
BASE_MODES = {"t2va", "i2va", "fl2va", "l2va"}
ALL_MODES = ("auto", "t2va", "i2va", "fl2va", "l2va", "ref2va")


def _read_required(path: Path) -> str:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise FileNotFoundError(f"H3 skill file is unavailable: {path}") from error
    if not value:
        raise ValueError(f"H3 skill file is empty: {path}")
    return value


def load_h3_system_prompt(skill_dir: Path, mode: str) -> str:
    """Load the Codex H3 skill as a portable system prompt for Gemma."""
    normalized_mode = mode.lower()
    if normalized_mode not in ALL_MODES:
        raise ValueError(f"unsupported H3 mode: {mode}")

    skill = _read_required(skill_dir / "SKILL.md")
    guides: list[tuple[str, str]] = []
    if normalized_mode in BASE_MODES or normalized_mode == "auto":
        guides.append(("base-en.txt", _read_required(skill_dir / "references" / "base-en.txt")))
    if normalized_mode == "ref2va" or normalized_mode == "auto":
        guides.append(("ref-en.txt", _read_required(skill_dir / "references" / "ref-en.txt")))

    guide_text = "\n\n".join(
        f"<reference name=\"{name}\">\n{contents}\n</reference>"
        for name, contents in guides
    )
    return f"""You are executing the MiniMax H3 prompt-writing skill included below.
Follow it exactly. The user supplies an H3 mode, duration, request, and optional
ordered reference images. Select the matching guide, preserve its exact field
names and section order, and ensure all shot timing fits the requested duration.
Return only the finished H3 prompt as plain text. Do not add Markdown fences,
analysis, prefaces, explanations, or follow-up questions.

<skill>
{skill}
</skill>

{guide_text}
"""


def image_data_url(path: Path) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"reference image not found: {resolved}")
    mime_type = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
    if not mime_type.startswith("image/"):
        raise ValueError(f"reference asset is not an image: {resolved}")
    encoded = base64.b64encode(resolved.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_user_content(
    request: str,
    mode: str,
    duration: float,
    images: list[Path],
) -> list[dict[str, Any]]:
    normalized_request = request.strip()
    if not normalized_request:
        raise ValueError("the prompt request cannot be empty")
    if not 4 <= duration <= 15:
        raise ValueError("MiniMax H3 duration must be between 4 and 15 seconds")

    text = (
        f"H3 mode: {mode.upper()}\n"
        f"Target duration: {duration:.2f} seconds\n"
        f"Reference images: {len(images)}; their order defines <Picture 1>, "
        "<Picture 2>, and subsequent labels.\n\n"
        f"User request:\n{normalized_request}"
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for index, image in enumerate(images, start=1):
        content.append({"type": "text", "text": f"<Picture {index}> follows."})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_data_url(image)},
            }
        )
    return content


def _message_text(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError("Gemma returned an invalid chat-completion response") from error
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}
        ]
        return "".join(parts).strip()
    raise RuntimeError("Gemma returned an unsupported message content type")


class H3GemmaClient:
    def __init__(
        self,
        api_url: str = DEFAULT_API_URL,
        model: str = "auto",
        timeout: float = 600,
        retries: int = 1,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.retries = retries

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.api_url}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemma API HTTP {error.code}: {body}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Gemma API unavailable at {self.api_url}: {error.reason}") from error
        if not isinstance(value, dict):
            raise RuntimeError("Gemma API returned a non-object response")
        return value

    def resolve_model(self) -> str:
        if self.model != "auto":
            return self.model
        payload = self._request("GET", "/models")
        models = payload.get("data") or payload.get("models") or []
        if not models:
            raise RuntimeError("Gemma server reported no loaded model")
        first = models[0]
        if not isinstance(first, dict):
            raise RuntimeError("Gemma server returned an invalid model record")
        model = first.get("id") or first.get("model") or first.get("name")
        if not model:
            raise RuntimeError("could not determine the loaded Gemma model name")
        self.model = str(model)
        return self.model

    def write_prompt(
        self,
        system_prompt: str,
        user_content: list[dict[str, Any]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        payload = {
            "model": self.resolve_model(),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                output = _message_text(self._request("POST", "/chat/completions", payload))
                if not output:
                    raise RuntimeError("Gemma returned an empty H3 prompt")
                return output
            except RuntimeError as error:
                last_error = error
                if attempt < self.retries:
                    time.sleep(1.5 * (attempt + 1))
        assert last_error is not None
        raise last_error


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a MiniMax H3 prompt with local Gemma using the Codex H3 skill"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--request", help="natural-language video request")
    source.add_argument("--request-file", type=Path, help="UTF-8 file containing the request")
    parser.add_argument("--mode", choices=ALL_MODES, default="auto")
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--image", type=Path, action="append", default=[],
                        help="ordered reference image; repeat for multiple pictures")
    parser.add_argument("--output", type=Path, help="write the generated prompt to this UTF-8 file")
    parser.add_argument("--skill-dir", type=Path,
                        default=Path(os.environ.get("H3_PROMPT_SKILL_DIR", DEFAULT_SKILL_DIR)))
    parser.add_argument("--api-url", default=os.environ.get("GEMMA_API_URL", DEFAULT_API_URL))
    parser.add_argument("--model", default="auto")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--print-system-prompt", action="store_true",
                        help="print the assembled skill prompt without calling Gemma")
    args = parser.parse_args(argv)
    if not 0 <= args.temperature <= 2:
        parser.error("--temperature must be between 0 and 2")
    if args.max_tokens < 256:
        parser.error("--max-tokens must be at least 256")
    if args.timeout <= 0 or args.retries < 0:
        parser.error("timeout must be positive and retries cannot be negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    request = args.request
    if request is None:
        request = args.request_file.read_text(encoding="utf-8")
    system_prompt = load_h3_system_prompt(args.skill_dir.expanduser().resolve(), args.mode)
    if args.print_system_prompt:
        print(system_prompt)
        return 0

    content = build_user_content(request, args.mode, args.duration, args.image)
    client = H3GemmaClient(args.api_url, args.model, args.timeout, args.retries)
    print(f"Gemma model: {client.resolve_model()}", file=sys.stderr)
    output = client.write_prompt(
        system_prompt,
        content,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    if args.output:
        resolved_output = args.output.expanduser().resolve()
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        resolved_output.write_text(output.rstrip() + "\n", encoding="utf-8")
        print(resolved_output)
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
