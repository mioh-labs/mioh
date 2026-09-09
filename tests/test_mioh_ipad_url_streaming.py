import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "apps" / "MiohRemote" / "MiohRemote"
PROJECT = ROOT / "apps" / "MiohRemote" / "MiohRemote.xcodeproj" / "project.pbxproj"
RESOLVER = APP / "IPadMediaURLResolver.swift"
BROWSER = APP / "IPadInteractiveMediaBrowser.swift"
DISCOVERY = APP / "IPadWebMediaDiscovery.swift"
INFO = APP / "Info.plist"
HARNESS = ROOT / "tests" / "swift" / "IPadMediaURLResolverHarness.swift"


def _iso_box(box_type, payload=b""):
    body_size = 8 + len(payload)
    return body_size.to_bytes(4, "big") + box_type.encode("ascii") + payload


class _ResolverFixtureHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    MASTER = b"""#EXTM3U
#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="audio-main",NAME="Japanese",LANGUAGE="ja",DEFAULT=YES,AUTOSELECT=YES,CHANNELS="2",URI="audio/japanese.m3u8?token=a%2Bb"
#EXT-X-STREAM-INF:BANDWIDTH=1000,RESOLUTION=640x360
low/index.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=5000,AVERAGE-BANDWIDTH=4500,RESOLUTION=1920x1080,FRAME-RATE=29.970,CODECS="avc1.640028,mp4a.40.2",AUDIO="audio-main"
high/index.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=15000,RESOLUTION=3840x2160
ultra/index.m3u8
"""
    HIGH_MEDIA = b"""#EXTM3U
#EXT-X-MEDIA-SEQUENCE:41
#EXT-X-MAP:URI="../assets/media.mp4",BYTERANGE="4@0"
#EXTINF:2.5,
#EXT-X-BYTERANGE:3@4
../assets/media.mp4
#EXTINF:1.5,
#EXT-X-BYTERANGE:2
../assets/media.mp4
#EXT-X-ENDLIST
"""
    DISCONTINUITY_MEDIA = b"""#EXTM3U
#EXT-X-MEDIA-SEQUENCE:500
#EXT-X-DISCONTINUITY-SEQUENCE:7
#EXTINF:2.0,
first.ts
#EXT-X-DISCONTINUITY
#EXTINF:3.0,
second.ts
#EXT-X-DISCONTINUITY
#EXTINF:4.0,
third.ts
#EXT-X-ENDLIST
"""
    LOW_MEDIA = b"""#EXTM3U
#EXTINF:4,
low.ts
#EXT-X-ENDLIST
"""
    HTML = b"""<!doctype html>
<html><body>
<video src="/fallback.mp4">
  <source src="stream\\/live.m3u8?token=a&amp;b=c">
</video>
</body></html>
"""
    HTML_MEDIA = b"""#EXTM3U
#EXTINF:1,
../chunks/a.ts
#EXT-X-ENDLIST
"""
    UNLABELED_RELAY = b"""#EXTM3U
# relays to a player-generated master playlist
inner/master.m3u8
"""
    RELAY_MASTER = b"""#EXTM3U
#EXT-X-STREAM-INF:BANDWIDTH=1000,RESOLUTION=640x360
low.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=2000,RESOLUTION=1280x720
high.m3u8
"""
    RELAY_MEDIA = b"""#EXTM3U
#EXTINF:5,
segment.ts
#EXT-X-ENDLIST
"""
    NESTED_AUDIO_OUTER_MASTER = b"""#EXTM3U
#EXT-X-STREAM-INF:BANDWIDTH=5000,RESOLUTION=1920x1080
inner/master.m3u8
"""
    NESTED_AUDIO_INNER_MASTER = b"""#EXTM3U
#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="inner-audio",NAME="Japanese",DEFAULT=YES,AUTOSELECT=YES,URI="audio.m3u8"
#EXT-X-STREAM-INF:BANDWIDTH=4500,RESOLUTION=1920x1080,CODECS="avc1.640028,mp4a.40.2",AUDIO="inner-audio"
video.m3u8
"""
    VARIANT_SWITCH_OUTER_MASTER = b"""#EXTM3U
#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="lower-audio",NAME="Lower Japanese",DEFAULT=YES,AUTOSELECT=YES,URI="audio/lower.m3u8"
#EXT-X-STREAM-INF:BANDWIDTH=6000000,AVERAGE-BANDWIDTH=5500000,RESOLUTION=1920x1080,FRAME-RATE=29.970,CODECS="avc1.640032,mp4a.40.2"
1080/master.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=3000000,AVERAGE-BANDWIDTH=2750000,RESOLUTION=1280x720,FRAME-RATE=29.970,CODECS="avc1.64001f,mp4a.40.2",AUDIO="lower-audio"
720/index.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=1000000,RESOLUTION=854x480,CODECS="avc1.64001e,mp4a.40.2"
480/index.m3u8
"""
    VARIANT_SWITCH_INNER_MASTER = b"""#EXTM3U
#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="inner-audio",NAME="Inner Japanese",DEFAULT=YES,AUTOSELECT=YES,URI="audio.m3u8"
#EXT-X-STREAM-INF:BANDWIDTH=5900000,AVERAGE-BANDWIDTH=5400000,RESOLUTION=1920x1080,FRAME-RATE=29.970,CODECS="avc1.640032,mp4a.40.2",AUDIO="inner-audio"
video.m3u8
"""
    VARIANT_SWITCH_MEDIA = b"""#EXTM3U
#EXT-X-TARGETDURATION:4
#EXTINF:4,
segment.ts
#EXT-X-ENDLIST
"""
    PARENT_AUDIO_NESTED_OUTER_MASTER = b"""#EXTM3U
#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="parent-audio",NAME="Parent Japanese",DEFAULT=YES,AUTOSELECT=YES,URI="parent/audio.m3u8"
#EXT-X-STREAM-INF:BANDWIDTH=6000000,RESOLUTION=1920x1080,CODECS="avc1.640032,mp4a.40.2",AUDIO="parent-audio"
inner/master.m3u8
"""
    PARENT_AUDIO_NESTED_INNER_MASTER = b"""#EXTM3U
#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="inner-audio",NAME="Inner Commentary",DEFAULT=YES,AUTOSELECT=YES,URI="audio/inner.m3u8"
#EXT-X-STREAM-INF:BANDWIDTH=5900000,RESOLUTION=1920x1080,CODECS="avc1.640032,mp4a.40.2"
1080.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=3000000,RESOLUTION=1280x720,CODECS="avc1.64001f,mp4a.40.2"
720.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=1000000,RESOLUTION=854x480,CODECS="avc1.64001e,mp4a.40.2"
480.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=500000,RESOLUTION=640x360,CODECS="avc1.4d401e,mp4a.40.2",AUDIO="inner-audio"
360.m3u8
"""
    AD_SELECTION_PAGE = b"""<!doctype html>
<html><body><video>
<source src="/selection/lead-in.m3u8">
<source src="/selection/programme.m3u8">
</video></body></html>
"""
    AD_SELECTION_LEAD_IN = b"""#EXTM3U
#EXT-X-TARGETDURATION:6
#EXTINF:6,
lead-1.ts
#EXTINF:6,
lead-2.ts
#EXTINF:6,
lead-3.ts
#EXT-X-ENDLIST
"""
    AD_SELECTION_PROGRAMME = b"""#EXTM3U
#EXT-X-TARGETDURATION:300
#EXTINF:300,
programme-1.ts
#EXTINF:300,
programme-2.ts
#EXT-X-ENDLIST
"""
    LAYER_ONE = b"""<!doctype html>
<html><body>
<iframe src="/layers/root"></iframe>
<object data="/layers/two"></object>
</body></html>
"""
    LAYER_TWO = b"""<!doctype html>
<html><body><embed data-lazy-src="/layers/three"></body></html>
"""
    LAYER_THREE = b"""<!doctype html>
<html><body><script>
window.player = {"player_url": "\\/layers\\/final.m3u8"};
</script></body></html>
"""
    LAYER_MEDIA = b"""#EXTM3U
#EXTINF:1,
segment.ts
#EXT-X-ENDLIST
"""
    DEPTH_ROOT = b"""<!doctype html>
<html><body>
<video src="/depth/fallback.mp4"></video>
<iframe src="/depth/one"></iframe>
</body></html>
"""
    DEPTH_ONE = b'<html><body><object data="/depth/two"></object></body></html>'
    DEPTH_TWO = b'<html><body><embed src="/depth/three"></body></html>'
    DEPTH_THREE = b'<html><body><iframe src="/depth/four"></iframe></body></html>'
    DEPTH_FOUR = b'<html><body><source src="/depth/late.m3u8"></body></html>'
    BODY_BUDGET_ROOT = (
        b'<html><body><iframe src="/body-budget/one"></iframe>'
        + (b" " * 700)
        + b"</body></html>"
    )
    BODY_BUDGET_ONE = (
        b'<html><body><iframe src="/body-budget/two"></iframe>'
        + (b" " * 700)
        + b"</body></html>"
    )
    DRM_MEDIA = b"""#EXTM3U
#EXT-X-KEY:METHOD=AES-128,URI="key.bin"
#EXTINF:1,
encrypted.ts
#EXT-X-ENDLIST
"""
    SAMPLE_AES_MEDIA = b"""#EXTM3U
#EXT-X-KEY:METHOD=SAMPLE-AES,URI="key.bin"
#EXTINF:1,
encrypted.ts
#EXT-X-ENDLIST
"""
    INVALID_EXTINF_INFINITY = b"""#EXTM3U
#EXTINF:inf,
segment.ts
#EXT-X-ENDLIST
"""
    INVALID_EXTINF_ZERO = b"""#EXTM3U
#EXTINF:0,
segment.ts
#EXT-X-ENDLIST
"""
    INVALID_EXTINF_OVERFLOW = b"""#EXTM3U
#EXTINF:9223372037,
segment.ts
#EXT-X-ENDLIST
"""
    DUPLICATE_MEDIA_SEQUENCE = b"""#EXTM3U
#EXT-X-MEDIA-SEQUENCE:4
#EXT-X-MEDIA-SEQUENCE:5
#EXTINF:1,
segment.ts
#EXT-X-ENDLIST
"""
    LATE_MEDIA_SEQUENCE = b"""#EXTM3U
#EXT-X-MEDIA-SEQUENCE:4
#EXTINF:1,
first.ts
#EXT-X-MEDIA-SEQUENCE:5
#EXTINF:1,
second.ts
#EXT-X-ENDLIST
"""
    MEDIA_SEQUENCE_OVERFLOW = b"""#EXTM3U
#EXT-X-MEDIA-SEQUENCE:9223372036854775806
#EXTINF:1,
first.ts
#EXTINF:1,
second.ts
#EXT-X-ENDLIST
"""
    REDIRECT_MEDIA = b"""#EXTM3U
#EXTINF:1,
chunk.ts
#EXT-X-ENDLIST
"""
    CONTEXT_MEDIA = b"""#EXTM3U
#EXTINF:1,
segment.ts
#EXT-X-ENDLIST
"""
    COOKIE_MASTER = b"""#EXTM3U
#EXT-X-STREAM-INF:BANDWIDTH=1000,RESOLUTION=640x360
variant.m3u8
"""
    COOKIE_VARIANT = b"""#EXTM3U
#EXT-X-MAP:URI="init.mp4"
#EXTINF:1,
segment.m4s
#EXT-X-ENDLIST
"""
    CHALLENGE_NESTED_ROOT = b"""<!doctype html>
<html><body>
<iframe src="/challenge"></iframe>
<iframe src="/challenge-priority/missing"></iframe>
</body></html>
"""
    CHALLENGE_VARIANT_MASTER = b"""#EXTM3U
#EXT-X-STREAM-INF:BANDWIDTH=2000,RESOLUTION=1280x720
../challenge
#EXT-X-STREAM-INF:BANDWIDTH=1000,RESOLUTION=640x360
missing.m3u8
"""
    HEAD_FALLBACK_MEDIA = b"""#EXTM3U
#EXTINF:4,
segment.ts
#EXT-X-ENDLIST
"""
    CHALLENGED_RENDITION_MASTER = b"""#EXTM3U
#EXT-X-STREAM-INF:BANDWIDTH=2000,RESOLUTION=1280x720
../challenge
#EXT-X-STREAM-INF:BANDWIDTH=1000,RESOLUTION=640x360
good.m3u8
"""
    MEDIA_BYTES = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    TS_MEDIA = b"".join(
        bytes([0x47]) + bytes([0x10 + packet_index]) * 187
        for packet_index in range(5)
    )
    AES128_KEY = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
    AES128_ENCRYPTED_TS = bytes.fromhex(
        "df5b43499913e2f89f3876694b5aba02746bbf9eeab7e5db524ec0f071d1d30e"
        "e0fa4f7ee2b371c3f3548bc2a2d90eefdaf8297052282226cd661b96d9517e6e"
        "6d20480a7a1b5ee2239d654ccccd0b759644038585c911f4a9ddcaf9a243771b"
        "3714f8a66258871c4c68897a818752aba3388f231c7e50da9bfe92447e152238"
        "36b260f4887038584e343afced4a849d59e9d04d3edec1e4469a82d52924dad2"
        "6b3dc96dacf8bd8e4f9402030834ef09c2bf3fc9df1ec935525a8fb575d62bd73"
        "3af4f8e1a767d98d1b3497c66fdadbb9dd9e9632bdd18d745d73fb1126be5fdd"
        "731efe0e7fc537f7dd73e698d13d10daaa484b104eadf6cc282ae462b925dea6"
        "2e4aa4631c8daedbb76f94fe5bc3e32bb6110bd9fe7750fdda905f0f6fcdf285"
        "063a033c3f28cdadf6fdda80995e1cb7a6db829ff93b61cc6302f1e69cdd1d96"
        "77aa049b7176738c0c2ad652270967d4aa2dc7958cd1eb3e4b98fee1afdc05e0c"
        "8ac82d181b4e9f01889105201b3e22ed3a585b15de8cee5ec4ca00387ddc3bf9"
        "44da099868d75202137874dc267bba57812e8825b3b7102d2a994b108ae36cdd8"
        "f882c6ca1c6a8ec632ca7cf9b22d54e013bc33221c0b96b6260e25f5602246d3"
        "d3f3bd5cb8fea6bf6cfbb7fc563b65315092b5991557bdce5327841fba22f8ea"
        "325d109dfe821b38f8bc14bfd9c0b9d11910122810f4b4ae7375bbd159a7009e"
        "2def0e77f96846802233af8ee77c9678182c2ce5972de4cdad69bead483266236"
        "60cb2542c61552fb9da06eac09a9f213fbc1a8cf7c588a16f26060dc0b490b7c"
        "167de7dce03a876beb3184526d2ef3cb8aaaaba1d4f664ac2906bdc735f487f75"
        "b74a2fc1dfe8255f9cd592b3ab45cc3e8e8ad716d3b44e6619f38ecc4d77b13e"
        "d1612c561f2defbcdca4acf3df77e8fd3bed42c7c7ea072999b07767c6680d83c"
        "09bc1966f1a2ef23d07de1e0cda0715bdeed86922dabc63d610e405b0b2b09bb"
        "c6b8a05a1230c1b13b5fc893fd8c2e8f6c43932aaff8f414ed9cc0d254660eb6"
        "b58967fe14393ea1d10042b2c60bb3ce0f6c6cde60d264d757d1c641ec9cf600"
        "bdf802440964351a57b5e94eb65847c301d9f5785c5e9d118fb72319480ea0eb8"
        "bd849903c15e61b5215a641c1ffaa4513e5529f7cbb790b0d04366ca1f0b608e"
        "7556e7fda18505be0c2f6524509b5fbf5ae027eaa326d3b2769517e6455616d4"
        "9cace37eee166d53ccf69c0ecc9e929d0c2941edec9942c43ce1c1fcef3982f56"
        "dd733c36fc59e0c5f81e34e0571c68ec439f693ce21870dd2a3ad7834c748bfe"
        "5013b36916a12ff902955078"
    )
    PNG_LIKE_PREFIX = b"\x89PNG\r\n\x1a\n" + (b"\x00" * (230 - 8))
    PNG_PREFIXED_TS_MEDIA = PNG_LIKE_PREFIX + TS_MEDIA
    FMP4_INIT = _iso_box("ftyp", b"isom") + _iso_box("moov", b"init")
    FMP4_FRAGMENT = (
        _iso_box("styp", b"msdh")
        + _iso_box("moof", b"fragment")
        + _iso_box("mdat", b"media")
    )
    SELF_CONTAINED_MP4 = (
        _iso_box("ftyp", b"isom")
        + _iso_box("moov", b"complete")
        + _iso_box("mdat", b"media")
    )

    def do_HEAD(self):
        self._dispatch(include_body=False)

    def do_GET(self):
        self._dispatch(include_body=True)

    def log_message(self, _format, *args):
        del args

    def _record(self):
        with self.server.request_lock:
            self.server.requests.append(
                (self.command, self.path, self.headers.get("Range"))
            )
            self.server.request_headers.append(
                (self.command, self.path, dict(self.headers.items()))
            )

    def _dispatch(self, include_body):
        self._record()
        path = urlsplit(self.path).path
        if path == "/go":
            self._redirect("/redirected/media.m3u8")
            return
        if path == "/redirect-private":
            port = self.server.server_address[1]
            self._redirect(f"http://localhost:{port}/layers/private-trap")
            return
        if path.startswith("/redirect-loop/"):
            try:
                generation = int(path.rsplit("/", 1)[1])
            except ValueError:
                generation = 0
            self._redirect(f"/redirect-loop/{generation + 1}")
            return
        if path == "/slow":
            if not include_body:
                self._send_bytes(b"xx", "application/octet-stream", include_body=False)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", "2")
            self.end_headers()
            try:
                self.wfile.write(b"x")
                self.wfile.flush()
                time.sleep(3)
                self.wfile.write(b"x")
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        if path in {"/shared/segment.ts", "/shared/cancel.ts"}:
            time.sleep(0.25)
            self._send_bytes(b"shared-media", "video/mp2t", include_body)
            return

        if path == "/rate-limit/first.ts":
            self.send_response(429)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Retry-After", "1")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        if path == "/rate-limit/second.ts":
            self._send_bytes(b"after-cooldown", "video/mp2t", include_body)
            return

        if path in {"/context/master.m3u8", "/context/segment.ts"}:
            valid_context = (
                self.headers.get("Cookie") == "mioh_session=allowed"
                and self.headers.get("User-Agent") == "MiohContextHarness/1.0"
                and self.headers.get("Referer", "").endswith("/watch/page")
                and "?" not in self.headers.get("Referer", "")
            )
            if not valid_context:
                self._send_bytes(
                    b"missing request context",
                    "text/plain",
                    include_body,
                    status=403,
                )
                return
            if path == "/context/master.m3u8":
                self._send_bytes(
                    self.CONTEXT_MEDIA,
                    "application/vnd.apple.mpegurl",
                    include_body,
                )
            else:
                self._send_bytes(b"context-ok", "video/mp2t", include_body)
            return

        if path.startswith("/variant-switch/"):
            valid_context = (
                "variant_session=allowed" in self.headers.get("Cookie", "")
                and self.headers.get("User-Agent") == "MiohVariantHarness/1.0"
                and self.headers.get("Referer", "").endswith("/watch/page")
                and "?" not in self.headers.get("Referer", "")
            )
            if not valid_context:
                self._send_bytes(
                    b"missing variant request context",
                    "text/plain",
                    include_body,
                    status=403,
                )
                return

        if path == "/challenge":
            self.send_response(403)
            self.send_header("Content-Type", "text/html")
            self.send_header("cf-mitigated", "challenge")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        if not include_body and path == "/head-fallback/challenge.m3u8":
            self.send_response(403)
            self.send_header("Content-Type", "text/html")
            self.send_header("cf-mitigated", "challenge")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        if not include_body and path == "/head-fallback/failure.m3u8":
            self._send_bytes(
                b"HEAD is not supported",
                "text/plain",
                include_body=False,
                status=405,
            )
            return

        if path == "/cookie-rotate/start":
            self.send_response(302)
            self.send_header("Location", "/cookie-rotate/master.m3u8")
            self.send_header(
                "Set-Cookie", "redirect_token=redirected; Path=/cookie-rotate"
            )
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        if path.startswith("/cookie-rotate/"):
            cookie = self.headers.get("Cookie", "")
            context_headers_valid = (
                self.headers.get("User-Agent") == "MiohCookieUpdateHarness/1.0"
                and self.headers.get("Referer", "").endswith("/watch/page")
                and "?" not in self.headers.get("Referer", "")
            )
            required = {
                "/cookie-rotate/master.m3u8": ["redirect_token=redirected"],
                "/cookie-rotate/variant.m3u8": [
                    "redirect_token=redirected",
                    "master_token=mastered",
                ],
                "/cookie-rotate/init.mp4": ["variant_token=variant"],
                "/cookie-rotate/segment.m4s": [
                    "variant_token=variant",
                    "init_token=initialized",
                ],
            }.get(path, [])
            stale_cookie_leaked = (
                path != "/cookie-rotate/master.m3u8"
                and "stale_token=remove-me" in cookie
            )
            if (
                not context_headers_valid
                or stale_cookie_leaked
                or not all(value in cookie for value in required)
            ):
                self._send_bytes(
                    b"missing updated cookie",
                    "text/plain",
                    include_body,
                    status=403,
                )
                return
            if path == "/cookie-rotate/master.m3u8":
                body, content_type = (
                    self.COOKIE_MASTER,
                    "application/vnd.apple.mpegurl",
                )
                set_cookie = "master_token=mastered; Path=/cookie-rotate"
            elif path == "/cookie-rotate/variant.m3u8":
                body, content_type = (
                    self.COOKIE_VARIANT,
                    "application/vnd.apple.mpegurl",
                )
                set_cookie = "variant_token=variant; Path=/cookie-rotate"
            elif path == "/cookie-rotate/init.mp4":
                body, content_type = b"INIT", "video/mp4"
                set_cookie = "init_token=initialized; Path=/cookie-rotate"
            elif path == "/cookie-rotate/segment.m4s":
                body, content_type = b"MEDIA", "video/mp4"
                set_cookie = None
            else:
                self._send_bytes(b"not found", "text/plain", include_body, status=404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            if set_cookie:
                self.send_header("Set-Cookie", set_cookie)
            if path == "/cookie-rotate/master.m3u8":
                self.send_header(
                    "Set-Cookie",
                    "stale_token=; Max-Age=0; Path=/cookie-rotate",
                )
            self.end_headers()
            if include_body:
                self.wfile.write(body)
            return

        if path == "/layers/root":
            port = self.server.server_address[1]
            body = f"""<!doctype html>
<html><body>
<video src="/layers/fallback.mp4"></video>
<iframe src="http://localhost:{port}/layers/private-trap"></iframe>
<iframe src="/layers/one"></iframe>
</body></html>
""".encode()
            self._send_bytes(body, "text/html; charset=utf-8", include_body)
            return

        fixtures = {
            "/hls/master.m3u8": (self.MASTER, "application/vnd.apple.mpegurl"),
            "/hls/high/index.m3u8": (
                self.HIGH_MEDIA,
                "application/vnd.apple.mpegurl",
            ),
            "/hls/discontinuity.m3u8": (
                self.DISCONTINUITY_MEDIA,
                "application/vnd.apple.mpegurl",
            ),
            "/hls/low/index.m3u8": (
                self.LOW_MEDIA,
                "application/vnd.apple.mpegurl",
            ),
            "/hls/ultra/index.m3u8": (
                self.HIGH_MEDIA,
                "application/vnd.apple.mpegurl",
            ),
            "/embed/page": (self.HTML, "text/html; charset=utf-8"),
            "/embed/stream/live.m3u8": (
                self.HTML_MEDIA,
                "application/vnd.apple.mpegurl",
            ),
            "/relay/outer.m3u8": (
                self.UNLABELED_RELAY,
                "application/vnd.apple.mpegurl",
            ),
            "/relay/inner/master.m3u8": (
                self.RELAY_MASTER,
                "application/vnd.apple.mpegurl",
            ),
            "/relay/inner/high.m3u8": (
                self.RELAY_MEDIA,
                "application/vnd.apple.mpegurl",
            ),
            "/relay/inner/low.m3u8": (
                self.RELAY_MEDIA,
                "application/vnd.apple.mpegurl",
            ),
            "/relay/inner/segment.ts": (b"relay", "video/mp2t"),
            "/nested-audio/outer.m3u8": (
                self.NESTED_AUDIO_OUTER_MASTER,
                "application/vnd.apple.mpegurl",
            ),
            "/nested-audio/inner/master.m3u8": (
                self.NESTED_AUDIO_INNER_MASTER,
                "application/vnd.apple.mpegurl",
            ),
            "/nested-audio/inner/video.m3u8": (
                self.RELAY_MEDIA,
                "application/vnd.apple.mpegurl",
            ),
            "/nested-audio/inner/audio.m3u8": (
                self.RELAY_MEDIA,
                "application/vnd.apple.mpegurl",
            ),
            "/variant-switch/master.m3u8": (
                self.VARIANT_SWITCH_OUTER_MASTER,
                "application/vnd.apple.mpegurl",
            ),
            "/variant-switch/1080/master.m3u8": (
                self.VARIANT_SWITCH_INNER_MASTER,
                "application/vnd.apple.mpegurl",
            ),
            "/variant-switch/1080/video.m3u8": (
                self.VARIANT_SWITCH_MEDIA,
                "application/vnd.apple.mpegurl",
            ),
            "/variant-switch/1080/audio.m3u8": (
                self.VARIANT_SWITCH_MEDIA,
                "application/vnd.apple.mpegurl",
            ),
            "/variant-switch/720/index.m3u8": (
                self.VARIANT_SWITCH_MEDIA,
                "application/vnd.apple.mpegurl",
            ),
            "/variant-switch/480/index.m3u8": (
                self.VARIANT_SWITCH_MEDIA,
                "application/vnd.apple.mpegurl",
            ),
            "/variant-switch/audio/lower.m3u8": (
                self.VARIANT_SWITCH_MEDIA,
                "application/vnd.apple.mpegurl",
            ),
            "/parent-audio-nested/master.m3u8": (
                self.PARENT_AUDIO_NESTED_OUTER_MASTER,
                "application/vnd.apple.mpegurl",
            ),
            "/parent-audio-nested/inner/master.m3u8": (
                self.PARENT_AUDIO_NESTED_INNER_MASTER,
                "application/vnd.apple.mpegurl",
            ),
            "/parent-audio-nested/inner/1080.m3u8": (
                self.VARIANT_SWITCH_MEDIA,
                "application/vnd.apple.mpegurl",
            ),
            "/parent-audio-nested/inner/720.m3u8": (
                self.VARIANT_SWITCH_MEDIA,
                "application/vnd.apple.mpegurl",
            ),
            "/parent-audio-nested/inner/480.m3u8": (
                self.VARIANT_SWITCH_MEDIA,
                "application/vnd.apple.mpegurl",
            ),
            "/parent-audio-nested/inner/360.m3u8": (
                self.VARIANT_SWITCH_MEDIA,
                "application/vnd.apple.mpegurl",
            ),
            "/parent-audio-nested/parent/audio.m3u8": (
                self.VARIANT_SWITCH_MEDIA,
                "application/vnd.apple.mpegurl",
            ),
            "/parent-audio-nested/inner/audio/inner.m3u8": (
                self.VARIANT_SWITCH_MEDIA,
                "application/vnd.apple.mpegurl",
            ),
            "/selection/page": (
                self.AD_SELECTION_PAGE,
                "text/html; charset=utf-8",
            ),
            "/selection/lead-in.m3u8": (
                self.AD_SELECTION_LEAD_IN,
                "application/vnd.apple.mpegurl",
            ),
            "/selection/programme.m3u8": (
                self.AD_SELECTION_PROGRAMME,
                "application/vnd.apple.mpegurl",
            ),
            "/layers/one": (self.LAYER_ONE, "text/html; charset=utf-8"),
            "/layers/two": (self.LAYER_TWO, "text/html; charset=utf-8"),
            "/layers/three": (self.LAYER_THREE, "text/html; charset=utf-8"),
            "/layers/final.m3u8": (
                self.LAYER_MEDIA,
                "application/vnd.apple.mpegurl",
            ),
            "/layers/segment.ts": (b"layer", "video/mp2t"),
            "/layers/fallback.mp4": (b"fallback", "video/mp4"),
            "/layers/private-trap": (b"trap", "text/html; charset=utf-8"),
            "/depth/root": (self.DEPTH_ROOT, "text/html; charset=utf-8"),
            "/depth/one": (self.DEPTH_ONE, "text/html; charset=utf-8"),
            "/depth/two": (self.DEPTH_TWO, "text/html; charset=utf-8"),
            "/depth/three": (self.DEPTH_THREE, "text/html; charset=utf-8"),
            "/depth/four": (self.DEPTH_FOUR, "text/html; charset=utf-8"),
            "/depth/late.m3u8": (
                self.LAYER_MEDIA,
                "application/vnd.apple.mpegurl",
            ),
            "/depth/fallback.mp4": (b"fallback", "video/mp4"),
            "/body-budget/root": (
                self.BODY_BUDGET_ROOT,
                "text/html; charset=utf-8",
            ),
            "/body-budget/one": (
                self.BODY_BUDGET_ONE,
                "text/html; charset=utf-8",
            ),
            "/drm/encrypted.m3u8": (
                self.DRM_MEDIA,
                "application/vnd.apple.mpegurl",
            ),
            "/drm/key.bin": (self.AES128_KEY, "application/octet-stream"),
            "/drm/encrypted.ts": (
                self.AES128_ENCRYPTED_TS,
                "video/mp2t",
            ),
            "/drm/sample-aes.m3u8": (
                self.SAMPLE_AES_MEDIA,
                "application/vnd.apple.mpegurl",
            ),
            "/invalid/extinf-infinity.m3u8": (
                self.INVALID_EXTINF_INFINITY,
                "application/vnd.apple.mpegurl",
            ),
            "/invalid/extinf-zero.m3u8": (
                self.INVALID_EXTINF_ZERO,
                "application/vnd.apple.mpegurl",
            ),
            "/invalid/extinf-overflow.m3u8": (
                self.INVALID_EXTINF_OVERFLOW,
                "application/vnd.apple.mpegurl",
            ),
            "/invalid/duplicate-media-sequence.m3u8": (
                self.DUPLICATE_MEDIA_SEQUENCE,
                "application/vnd.apple.mpegurl",
            ),
            "/invalid/late-media-sequence.m3u8": (
                self.LATE_MEDIA_SEQUENCE,
                "application/vnd.apple.mpegurl",
            ),
            "/invalid/media-sequence-overflow.m3u8": (
                self.MEDIA_SEQUENCE_OVERFLOW,
                "application/vnd.apple.mpegurl",
            ),
            "/redirected/media.m3u8": (
                self.REDIRECT_MEDIA,
                "application/vnd.apple.mpegurl",
            ),
            "/challenge-priority/page": (
                self.CHALLENGE_NESTED_ROOT,
                "text/html; charset=utf-8",
            ),
            "/challenge-priority/master.m3u8": (
                self.CHALLENGE_VARIANT_MASTER,
                "application/vnd.apple.mpegurl",
            ),
            "/head-fallback/challenge.m3u8": (
                self.HEAD_FALLBACK_MEDIA,
                "application/vnd.apple.mpegurl",
            ),
            "/head-fallback/failure.m3u8": (
                self.HEAD_FALLBACK_MEDIA,
                "application/vnd.apple.mpegurl",
            ),
            "/variant-fallback/master.m3u8": (
                self.CHALLENGED_RENDITION_MASTER,
                "application/vnd.apple.mpegurl",
            ),
            "/variant-fallback/good.m3u8": (
                self.HEAD_FALLBACK_MEDIA,
                "application/vnd.apple.mpegurl",
            ),
            "/redirected/chunk.ts": (b"redirected", "video/mp2t"),
            "/embed/chunks/a.ts": (b"html", "video/mp2t"),
            "/fallback.mp4": (b"fallback", "video/mp4"),
            "/materialize/png-prefixed.bin": (
                self.PNG_PREFIXED_TS_MEDIA,
                "application/octet-stream",
            ),
            "/materialize/ordinary.ts": (self.TS_MEDIA, "video/mp2t"),
            "/materialize/disguised.mp4": (self.TS_MEDIA, "video/mp4"),
            "/materialize/init.mp4": (self.FMP4_INIT, "video/mp4"),
            "/materialize/fragment.m4s": (self.FMP4_FRAGMENT, "video/mp4"),
            "/materialize/self-contained.mp4": (
                self.SELF_CONTAINED_MP4,
                "video/mp4",
            ),
            "/materialize/mapped-transport.mp4": (self.TS_MEDIA, "video/mp4"),
        }
        if path == "/hls/assets/media.mp4":
            self._send_range_capable(self.MEDIA_BYTES, include_body)
            return
        fixture = fixtures.get(path)
        if fixture is None:
            self._send_bytes(b"not found", "text/plain", include_body, status=404)
            return
        self._send_bytes(fixture[0], fixture[1], include_body)

    def _redirect(self, location):
        self.send_response(302)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _send_bytes(self, body, content_type, include_body, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if include_body:
            self.wfile.write(body)

    def _send_range_capable(self, body, include_body):
        range_header = self.headers.get("Range")
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", range_header or "")
        if match:
            start, end = map(int, match.groups())
            if start >= len(body) or end < start:
                self._send_bytes(b"", "video/mp4", include_body, status=416)
                return
            end = min(end, len(body) - 1)
            selected = body[start : end + 1]
            self.send_response(206)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(selected)))
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(body)}")
            self.end_headers()
            if include_body:
                self.wfile.write(selected)
            return
        self._send_bytes(body, "video/mp4", include_body)


class _ResolverFixtureServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _ResolverFixtureHandler)
        self.request_lock = threading.Lock()
        self.requests = []
        self.request_headers = []

    def reset_requests(self):
        with self.request_lock:
            self.requests.clear()
            self.request_headers.clear()

    def request_snapshot(self):
        with self.request_lock:
            return list(self.requests)

    def request_header_snapshot(self):
        with self.request_lock:
            return list(self.request_headers)


class MiohIPadURLStreamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.resolver = RESOLVER.read_text() if RESOLVER.exists() else ""
        cls.browser = BROWSER.read_text() if BROWSER.exists() else ""
        cls.discovery = DISCOVERY.read_text() if DISCOVERY.exists() else ""
        cls.store = (APP / "IPadStandaloneStore.swift").read_text()
        cls.view = (APP / "IPadStandaloneView.swift").read_text()
        cls.project = PROJECT.read_text()
        cls.info = INFO.read_text()
        cls.runtime_directory = None
        cls.harness_binary = None
        cls.fixture_server = None

        if sys.platform != "darwin":
            return
        swiftc = shutil.which("swiftc")
        if not swiftc:
            xcrun = shutil.which("xcrun")
            if xcrun:
                swiftc = subprocess.check_output(
                    [xcrun, "--find", "swiftc"], text=True
                ).strip()
        if not swiftc:
            return

        cls.runtime_directory = tempfile.TemporaryDirectory(
            prefix="mioh-url-resolver-tests-"
        )
        cls.harness_binary = Path(cls.runtime_directory.name) / "resolver-harness"
        build = subprocess.run(
            [
                swiftc,
                "-D",
                "MIOH_TESTING",
                "-parse-as-library",
                str(RESOLVER),
                str(HARNESS),
                "-o",
                str(cls.harness_binary),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=120,
        )
        if build.returncode != 0:
            raise AssertionError(f"URL resolver harness did not compile:\n{build.stdout}")

        cls.fixture_server = _ResolverFixtureServer()
        cls.fixture_thread = threading.Thread(
            target=cls.fixture_server.serve_forever,
            name="mioh-url-resolver-fixture",
            daemon=True,
        )
        cls.fixture_thread.start()
        cls.fixture_base_url = (
            f"http://127.0.0.1:{cls.fixture_server.server_address[1]}/"
        )

    @classmethod
    def tearDownClass(cls):
        if cls.fixture_server:
            cls.fixture_server.shutdown()
            cls.fixture_server.server_close()
            cls.fixture_thread.join(timeout=2)
        if cls.runtime_directory:
            cls.runtime_directory.cleanup()

    def setUp(self):
        if self.fixture_server:
            self.fixture_server.reset_requests()

    def run_resolver_harness(self, scenario, timeout=10):
        if not self.harness_binary:
            self.skipTest("macOS Swift compiler is required for URL resolver integration tests")
        result = subprocess.run(
            [str(self.harness_binary), scenario, self.fixture_base_url],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"resolver harness scenario {scenario!r} failed:\n{result.stdout}",
        )
        return self.fixture_server.request_snapshot()

    def assert_contracts(self, source, contracts):
        for contract in contracts:
            self.assertIn(contract, source, f"missing source contract: {contract}")

    def test_resolver_contract_is_part_of_the_ios_target(self):
        self.assertTrue(RESOLVER.exists(), "IPadMediaURLResolver.swift is missing")
        self.assert_contracts(self.resolver, [
            "struct IPadResolvedMediaSource",
            "enum Kind",
            "case hls",
            "case progressive",
            "let submittedURL: URL",
            "let mediaURL: URL",
            "let contentType: String?",
            "enum IPadMediaURLResolverError",
            "struct IPadMediaURLResolver",
            "policy: IPadMediaURLResolutionPolicy = .userSubmitted",
            "static func normalizedHTTPURL(_ raw: String) -> URL?",
        ])
        self.assertIn("IPadMediaURLResolver.swift in Sources", self.project)

    def test_resolver_accepts_only_safe_http_media_urls(self):
        self.assert_contracts(self.resolver, [
            'scheme == "http" || scheme == "https"',
            "components.user == nil",
            "components.password == nil",
            "components.fragment = nil",
            '"m3u8"',
            '"mp4"',
            '"mov"',
            '"m4v"',
            '"application/vnd.apple.mpegurl"',
            '"application/x-mpegurl"',
            'hasPrefix("video/")',
            'hasPrefix("#EXTM3U")',
        ])

    def test_html_discovery_prefers_hls_and_resolves_relative_candidates(self):
        self.assert_contracts(self.resolver, [
            '"og:video"',
            '"video"',
            '"source"',
            'relativeTo:',
            '"&amp;"',
            '"\\/"',
            "response.url",
            "case .hls",
        ])
        self.assertRegex(
            self.resolver,
            r"(?s)(?:hlsCandidates|\.hls).*?(?:progressiveCandidates|\.progressive)",
        )

    def test_static_player_page_traversal_is_depth_and_request_bounded(self):
        self.assert_contracts(self.resolver, [
            "private static let maximumPageDepth = 3",
            "private static let maximumNestedPagesPerPage = 8",
            'for embedTag in ["iframe", "embed"]',
            'regexMatches("<object\\\\b[^>]*>", in: html)',
            '"data-lazy-src"',
            "var visitedPages = Set<URL>()",
            "budget: budget",
            "progressiveFallback",
            "isDisallowedDiscoveredLocalURL",
            "isSameOrigin",
        ])

    def test_network_analysis_is_bounded_and_cancellable(self):
        self.assert_contracts(self.resolver, [
            "URLSession",
            "Task.checkCancellation()",
            "maximumResponseBytes",
            "maximumRedirectCount",
            "maximumRequestCount",
            "maximumCumulativeResponseBytes",
            "resolutionTimeout",
            "deadline = Date().addingTimeInterval(timeout)",
            "deadline.timeIntervalSinceNow",
            "min(requestTimeout, remainingTimeout)",
            "maximumCumulativeResponseBytes",
            "consumeResponseBytes",
        ])

    def test_master_variant_is_bounded_for_ipad_restoration(self):
        self.assert_contracts(self.resolver, [
            'attributes["RESOLUTION"]',
            "IPadRestorationMediaLimits.accepts(",
            "maximumLongEdge = 1_920",
            "maximumShortEdge = 1_080",
            "let unknown = variants.filter",
            "$0.bandwidth < $1.bandwidth",
        ])
        self.assert_contracts(self.resolver, [
            "parseUnlabeledPlaylistReferences",
            "参照先playlist",
        ])
        self.run_resolver_harness("unlabeled-playlist-reference")

    def test_realtime_has_a_dedicated_1080p_t48_budget(self):
        self.assert_contracts(self.resolver, [
            "referenceClipLength = 18",
            "realtimeReferenceClipLength = 48",
            "maximumRealtimePixelFrames",
            "pixelFrameBudget: Int = maximumPixelFrames",
        ])
        self.run_resolver_harness("media-limits")

    def test_url_input_does_not_disable_transport_security_globally(self):
        self.assertNotIn("NSAllowsArbitraryLoads", self.info)

    def test_url_input_hands_resolved_media_to_realtime_playback(self):
        combined = self.store + self.view
        self.assert_contracts(combined, [
            "IPadMediaURLResolver",
            "IPadResolvedMediaSource",
            "TextField",
            '"URLを解析"',
            "selectedTab = .playback",
            "realtimePlayer",
        ])

        # HLS cannot be sent through the existing finite-file executeLocal /
        # AVAssetReader segment path. The UI must dispatch by the resolved kind.
        self.assertRegex(combined, r"(?s)switch\s+.*?\.kind.*?case \.hls:.*?case \.progressive:")

    def test_browser_selection_compares_playable_sources_before_accepting(self):
        self.assert_contracts(self.resolver, [
            "enum IPadBrowserMediaSourceSelector",
            "maximumPlayableChoices = 40",
            "static func isHighConfidenceAdvertisementSource(",
            'host == "saawsedge.com"',
            'host.hasSuffix(".saawsedge.com")',
            "static func preferredIndex(",
            "static func shouldAcceptImmediately(",
            "private static func shouldReplace(",
            "programmeDuration(",
            "currentDuration * 1.5",
            "IPadBrowserMediaEvidence",
            "resolvedHLSChoices",
            "preferredCollectedChoice()",
        ])
        self.assert_contracts(self.store, [
            "var resolvedChoices: [IPadResolvedMediaSource] = []",
            "var resolvedEvidence: [IPadBrowserMediaEvidence?] = []",
            "IPadBrowserMediaSourceSelector.deduplicationKey",
            "resolvedChoices.append(source)",
            "IPadBrowserMediaSourceSelector.preferredIndex(",
            "IPadBrowserMediaSourceSelector.shouldAcceptImmediately(",
            "IPadBrowserMediaSourceSelector.isHighConfidenceAdvertisementSource(",
            "右下の広告プレイヤーが発行したHLS候補を除外しました。",
            "期限切れ前に選択しました",
            "evidence: resolvedEvidence",
            "let selectedSource = resolvedChoices.remove(at: selectedIndex)",
            "isSkippableBrowserCandidateError",
            "firstUsefulError = firstUsefulError ?? error",
        ])
        selection_body = self.store.split(
            "func selectBrowserCandidates(", 1
        )[1].split("private func acceptResolvedURLSource", 1)[0]
        self.assertLess(
            selection_body.index("resolvedChoices.append(source)"),
            selection_body.index("if try await acceptCollectedChoices()"),
        )
        self.assertNotIn("if !resolvedChoices.isEmpty { break }", selection_body)
        self.run_resolver_harness("browser-selection")
        self.run_resolver_harness("html-ad-selection")

    def test_browser_candidate_failures_do_not_mask_later_sources(self):
        selection_body = self.store.split(
            "func selectBrowserCandidates(", 1
        )[1].split("private func acceptResolvedURLSource", 1)[0]
        self.assertIn("maximumPlayableChoices", selection_body)
        self.assertLess(
            selection_body.index("remainingAttempts -= 1"),
            selection_body.index("Self.resolveBrowserCandidate("),
        )
        self.assertIn("private nonisolated static func resolveBrowserCandidate", self.store)
        self.assertIn(
            "catch IPadMediaURLResolverError.unsafeInitialURL", selection_body
        )
        self.assertIn("remainingAttempts += 1", selection_body)
        self.assertIn("if !Self.isSkippableBrowserCandidateError(error)", selection_body)
        self.assertNotIn("} catch {\n            lastError = error", selection_body)
        self.assertRegex(
            selection_body,
            r"(?s)do\s*\{.*?acceptResolvedURLSource\(.*?"
            r"\}\s*catch is CancellationError.*?catch\s*\{",
        )
        self.assertRegex(
            selection_body,
            r"pendingInteractionError\s*\?\?\s*firstUsefulError",
        )

    def test_exhausted_live_browser_candidates_return_to_waiting_not_global_failure(self):
        selection_body = self.store.split(
            "func selectBrowserCandidates(", 1
        )[1].split("private func acceptResolvedURLSource", 1)[0]
        terminal_catch = selection_body.rsplit("} catch {", 1)[1]
        self.assert_contracts(terminal_catch, [
            "Browser candidates are a live observation",
            "state = .idle",
            "urlInputStatus = error.localizedDescription",
            "appendLog(",
            "level: .warning",
            "return false",
        ])
        self.assertNotIn("state = .failed", terminal_catch)

    def test_static_progressive_fallback_is_policy_checked(self):
        fallback = self.resolver.split(
            "if progressiveFallback == nil", 1
        )[1].split("for candidate in candidates.hlsCandidates", 1)[0]
        self.assertIn("Self.discoveredPolicy(for: candidate, on: page)", fallback)
        self.assertIn("Self.isURL(candidate, allowedBy: candidatePolicy)", fallback)

    def test_live_autostart_begins_near_the_recent_playlist_edge(self):
        self.assert_contracts(self.store, [
            "private let startupSegmentCount = 3",
            "playlist.segments.suffix(startupSegmentCount).first?.startSeconds",
            "$0.startSeconds + $0.duration > startSeconds",
        ])
        self.assertRegex(
            self.store,
            r"(?s)if requestedTarget == 0,.*?playlist\.isLive.*?"
            r"playlist\.segments\.suffix\(startupSegmentCount\)\.first\?\.startSeconds",
        )

    def test_authenticated_request_context_reaches_playlist_and_segments(self):
        self.run_resolver_harness("request-context")
        headers = self.fixture_server.request_header_snapshot()
        playlist_accepts = [
            values.get("Accept", "")
            for _, path, values in headers
            if urlsplit(path).path == "/context/master.m3u8"
        ]
        segment_accepts = [
            values.get("Accept", "")
            for _, path, values in headers
            if urlsplit(path).path == "/context/segment.ts"
        ]
        self.assertTrue(playlist_accepts)
        self.assertTrue(segment_accepts)
        self.assertTrue(all("mpegurl" in value.lower() for value in playlist_accepts))
        self.assertTrue(all("mpegurl" not in value.lower() for value in segment_accepts))
        self.assert_contracts(self.resolver, [
            "struct IPadMediaRequestCookie: Sendable, Equatable",
            "struct IPadMediaRequestContext: Sendable, Equatable",
            "func matches(_ url: URL",
            "expiresAt.map({ $0 > now })",
            "includesSubdomains && host.hasSuffix",
            "requestContext?.applying(to: &request)",
            "let requestContext: IPadMediaRequestContext?",
            'request.setValue(nil, forHTTPHeaderField: "Origin")',
            'request.setValue(origin.absoluteString, forHTTPHeaderField: "Origin")',
            "private static func sanitizedOrigin",
        ])
        self.assert_contracts(self.store, [
            "context: candidate.requestContext",
            "requestContext: source.requestContext",
            "AVURLAssetHTTPCookiesKey",
            "AVURLAssetHTTPUserAgentKey",
        ])
        self.assertNotIn("AVURLAssetHTTPHeaderFieldsKey", self.store)

    def test_hls_materialization_rejects_html_and_preserves_mp4_segments(self):
        self.assert_contracts(self.resolver, [
            "private static func looksLikeHTML",
            "HLS区間の応答が動画ではなくHTMLでした",
            "private static func materializedFileExtension",
            'resourceURL.pathExtension.lowercased()',
            '["ftyp", "styp", "moof", "moov"]',
            'return "mp4"',
            'return "ts"',
        ])

    def test_materializer_strips_png_prefix_before_mpeg_ts(self):
        requests = self.run_resolver_harness("png-prefixed-ts")
        paths = [urlsplit(path).path for _, path, _ in requests]
        self.assertEqual(
            paths,
            [
                "/materialize/png-prefixed.bin",
                "/materialize/ordinary.ts",
                "/materialize/disguised.mp4",
            ],
        )
        self.assert_contracts(self.resolver, [
            "private static func normalizedMediaSegmentData(",
            "private static func transportStreamOffset(in data: Data)",
            "let requiredPackets = 5",
            "return offset == 0 ? data : Data(data[offset...])",
        ])

    def test_materializer_uses_payload_container_before_ext_x_map_hint(self):
        requests = self.run_resolver_harness("mapped-container-normalization")
        paths = [urlsplit(path).path for _, path, _ in requests]
        self.assertEqual(
            paths,
            [
                "/materialize/init.mp4",
                "/materialize/fragment.m4s",
                "/materialize/self-contained.mp4",
                "/materialize/mapped-transport.mp4",
            ],
        )
        self.assert_contracts(self.resolver, [
            "let effectiveInitializationData: Data?",
            "containsISOBaseMediaInitialization(segmentData)",
            "HLS初期化データの応答が動画ではなくHTMLでした",
            "if transportStreamOffset(in: segmentData) == 0 { return \"ts\" }",
        ])

    def test_cloudflare_challenge_is_reported_as_interaction_required(self):
        self.run_resolver_harness("interaction-required")
        self.assert_contracts(self.resolver, [
            'value(forHTTPHeaderField: "cf-mitigated")',
            '== "challenge"',
            "case interactionRequired(URL?)",
            "interactionRequired(destinationURL ?? response.url)",
            "isInteractionChallengeURL",
            'host == "challenges.cloudflare.com"',
            'contains("/cdn-cgi/challenge-platform/")',
            "validateNoInteractionChallenge(payload.response)",
            "対話式のブラウザ確認が必要なため、このページは自動解析できません。",
        ])
        self.assert_contracts(self.store, [
            "IPadMediaURLResolver.interactionChallengeError(",
            "if let error = validate(httpResponse)",
        ])
        head_path = self.resolver.split("if let headPayload,", 1)[1].split(
            "let payload = try await request(", 1
        )[0]
        self.assertIn(
            "try Self.validateSuccessfulResponse(headPayload.response)",
            head_path,
        )

    def test_head_challenge_or_failure_falls_back_to_one_bounded_get(self):
        requests = self.run_resolver_harness("head-fallback")
        for path in [
            "/head-fallback/challenge.m3u8",
            "/head-fallback/failure.m3u8",
        ]:
            methods = [method for method, request_path, _ in requests if request_path == path]
            self.assertEqual(methods, ["HEAD", "GET"])

        resolve_entry = self.resolver.split("func resolve(", 1)[1].split(
            "private func resolveStaticMediaPages(", 1
        )[0]
        self.assertRegex(
            resolve_entry,
            r"(?s)catch is IPadMediaURLResolverError.*?headPayload = nil.*?"
            r'let payload = try await request\(\s*submittedURL,\s*method: "GET"',
        )

    def test_master_continues_after_one_challenged_rendition(self):
        requests = self.run_resolver_harness("challenged-rendition-fallback")
        paths = [urlsplit(path).path for _, path, _ in requests]
        self.assertIn("/challenge", paths)
        self.assertIn("/variant-fallback/good.m3u8", paths)
        self.assertLess(
            paths.index("/challenge"),
            paths.index("/variant-fallback/good.m3u8"),
        )

        variant_loop = self.resolver.split(
            "for (variantIndex, variant) in orderedVariants.enumerated()", 1
        )[1].split("let playlist = try Self.parseMediaPlaylist", 1)[0]
        self.assertIn(
            "catch IPadMediaURLResolverError.interactionRequired",
            variant_loop,
        )
        self.assertIn("pendingInteractionURL", variant_loop)
        self.assertNotIn(
            "throw IPadMediaURLResolverError.interactionRequired(challengedURL)",
            variant_loop,
        )

    def test_browser_interaction_returns_to_visible_page_not_challenged_resource(self):
        self.assert_contracts(self.discovery, [
            "let interactionPageURL: URL?",
            "interactionPageURL: URL? = nil",
            "self.interactionPageURL = interactionPageURL",
        ])
        self.assert_contracts(self.browser, [
            "let fallbackFrameURL = Self.sanitizedPublicHTTPSURL(webView.url)",
            "interactionPageURL: fallbackFrameURL",
        ])
        selection = self.store.split("func selectBrowserCandidates(", 1)[1].split(
            "private func acceptResolvedURLSource", 1
        )[0]
        interaction = selection.split(
            "catch IPadMediaURLResolverError.interactionRequired", 1
        )[1].split("} catch {", 1)[0]
        self.assert_contracts(interaction, [
            "candidate.interactionPageURL",
            "candidate.requestContext.referer",
            "candidate.url",
            "urlInteractionURL = urlInteractionURL ?? targetURL",
        ])
        self.assertNotIn("challengedURL", interaction)

    def test_public_discovered_policy_rejects_legacy_and_resolved_private_hosts(self):
        self.run_resolver_harness("public-url-policy")
        self.assert_contracts(self.resolver, [
            "resolvedAddressesArePublic(host)",
            "getaddrinfo(host, nil, &hints, &result)",
            "isPublicIPv4(ipv4)",
            "guard isPublicIPv6(bytes) else { return false }",
        ])

    def test_visible_browser_policy_accepts_hostname_vpn_fake_ip_for_hls_cdn(self):
        self.assert_contracts(self.resolver, [
            "case visibleBrowserDiscovered(URL)",
            "approvedOrigin: URL",
            "guard isCanonicalHTTPSOrigin(approvedOrigin)",
            "public HTTPS HLS",
            "CDN children",
            "allowsHostnameVPNBenchmarkTranslation: true",
            "!isIPAddressLiteral(host)",
            "isVPNBenchmarkIPv4(ipv4)",
            "first == 198 && (second == 18 || second == 19)",
        ])
        self.assert_contracts(self.store, [
            "let policy = browserResolutionPolicy(for: candidate)",
            "policy: policy",
            "candidate.selectionState == .activeCurrentSource",
            "evidence.isPlaying",
            "evidence.isVisible",
            "evidence.renderedArea >= 4_096",
            "else { return .publicDiscovered }",
            "let approvedOrigin = browserOriginURL(for: candidate.url)",
            "return .visibleBrowserDiscovered(approvedOrigin)",
        ])

    def test_browser_vpn_policy_downgrades_unobserved_cross_origin_children(self):
        self.assert_contracts(self.resolver, [
            "case .visibleBrowserDiscovered(let approvedOrigin):",
            "isSameOrigin(candidate, approvedOrigin)",
            "? .visibleBrowserDiscovered(approvedOrigin) : .publicDiscovered",
            "private static func isCanonicalHTTPSOrigin",
        ])

    def test_user_submitted_authority_is_not_transitive(self):
        self.assert_contracts(self.resolver, [
            "let effectivePolicy = try Self.initialRequestPolicy(",
            "policy: effectivePolicy",
            "resolutionPolicy: effectivePolicy",
            "private static func initialRequestPolicy(",
            "if isPublicHTTPSURL(submittedURL)",
            "return .publicDiscovered",
            "return .submittedPageSameOrigin(origin)",
        ])
        resolve_body = self.resolver.split("func resolve(", 1)[1].split(
            "private func resolveStaticMediaPages(", 1
        )[0]
        self.assertNotIn("resolutionPolicy: policy", resolve_body)

    def test_html_discoveries_do_not_inherit_unrestricted_user_policy(self):
        self.assert_contracts(self.resolver, [
            "case submittedPageSameOrigin(URL)",
            "initialDiscoveredPolicy(",
            "discoveredPolicy(for: candidate, on: page)",
            "policy: candidatePolicy",
            "resolutionPolicy: candidatePolicy",
            "isURL(",
            "allowedBy: entry.options.resolutionPolicy",
        ])
        static_body = self.resolver.split(
            "private func resolveStaticMediaPages(", maxsplit=1
        )[1].split("private static func progressiveSource(", maxsplit=1)[0]
        self.assertNotIn("policy: policy", static_body)

    def test_response_cookies_rotate_through_redirect_variant_init_and_segment(self):
        self.run_resolver_harness("cookie-updates")
        self.assert_contracts(self.resolver, [
            "private actor IPadMediaCookieJar",
            "HTTPCookie.cookies(",
            "await context?.updateCookies(from: payload.response)",
            "await requestContext?.updateCookies(from: payload.response)",
            "await entry.options.requestContext?.updateCookies(from: response)",
            "existingBroadDomains.contains(cookie.domain)",
        ])
        self.assert_contracts(self.store, [
            "await requestContext?.updateCookies(from: httpResponse)",
            "await requestContext?.updateCookies(from: response)",
        ])

    def test_challenge_error_survives_nested_and_variant_candidate_fallbacks(self):
        self.run_resolver_harness("challenge-priority")
        self.assertGreaterEqual(
            self.resolver.count(
                "catch IPadMediaURLResolverError.interactionRequired"
            ),
            3,
        )
        self.assertNotIn("var encounteredInteractionRequired", self.resolver)

    def test_cookie_replay_preserves_same_site_and_native_provenance(self):
        self.assert_contracts(self.resolver, [
            "enum IPadMediaCookieSameSitePolicy",
            "case strict",
            "case lax",
            "case none",
            ".sameSitePolicy",
            "cookieSourceURL",
            "allowsCredentialReplay",
            "allowsCrossSiteCredentialReplay",
            "crossSiteCredentialsAllowed",
        ])
        self.assert_contracts(self.browser, [
            "let remainingCookies = requestCookies.filter",
            "prioritizedCookies + remainingCookies",
            "Self.relaxedWebCompatibilityEnabled",
            "let credentialAuthorizedURLKeys = Set(",
            "state.isPlaying, state.isVisible, state.visibilityAttested",
            "state.renderedArea >= 4_096",
            "credentialAuthorizedURLKeys.contains(candidate.url.absoluteString)",
            "Self.isVerifiedPageHLSObservation(candidate.sourceKind)",
            "origin: Self.requiresBrowserOriginHeader(candidate.sourceKind)",
            "normalized.contains(\"fetch\") || normalized.contains(\"xhr\")",
        ])
        replay_gate = self.browser.split(
            "let allowsCrossSiteCredentialReplay =", 1
        )[1].split("return IPadWebMediaCandidate(", 1)[0]
        self.assertNotIn(
            "activeURLKeys.contains(candidate.url.absoluteString)", replay_gate
        )
        credential_helper = self.browser.split(
            "private static func isVerifiedPageHLSObservation", 1
        )[1].split("private static let blankPageHTML", 1)[0]
        self.assertNotIn("active-current-source", credential_helper)
        self.assertNotIn("page-fetch-hls-response", credential_helper)
        self.assertNotIn("page-xhr-hls-response", credential_helper)

    def test_live_prefetch_is_bounded_and_rebases_both_players_on_drop(self):
        self.assert_contracts(self.store, [
            "private actor IPadLiveHLSPrefetchBuffer",
            "maximumBufferedSegments",
            "maximumBufferedBytes: 192 * 1_024 * 1_024",
            "let maximumPrefetchCount = min(",
            "let desiredCount = min(startupSegmentCount, buffered.count)",
            "pendingRebase = makeRebase(target: target)",
            "let skippedDuration = missingCount * estimatedDuration",
            "private func rebaseLivePlayback(",
            "sourcePlayer.pause()",
            "restoredPlayer.pause()",
            "clearRestoredQueue(removingFiles: true)",
            "state = shouldPlay ? .followingLiveEdge : .paused",
        ])
        self.assertRegex(
            self.store,
            r"(?s)private func rebaseLivePlayback\(.*?sourcePlayer\.pause\(\).*?"
            r"restoredPlayer\.pause\(\).*?clearRestoredQueue\(removingFiles: true\).*?"
            r"prepareSourceSeek\(item: item, generation: generation\)",
        )

    def test_url_autostart_waits_until_resolution_finishes(self):
        self.assertRegex(
            self.view,
            r"(?s)let accepted = await store\.selectURLInput\(.*?"
            r"selectionOwnerID: selectionOwnerID.*?guard accepted else.*?"
            r"selectedTab = \.playback.*?tryAutoStartURLPlayback\(\)",
        )
        self.assertRegex(
            self.view,
            r"(?s)private func tryAutoStartURLPlayback\(\).*?"
            r"guard scenePhase == \.active, selectedTab == \.playback,\s*"
            r"autoStartURLPlayback,\s*"
            r"!store\.isResolvingURL",
        )

    def test_url_autostart_is_not_stopped_by_input_change_observation(self):
        self.assertNotIn(".onChange(of: store.inputURL)", self.view)
        analysis_body = self.view.split(
            "private func analyzeURLAndStartPlayback()", maxsplit=1
        )[1].split("private func tryAutoStartURLPlayback()", maxsplit=1)[0]
        after_resolution = analysis_body.split(
            "guard accepted else", maxsplit=1
        )[1]
        self.assertNotIn("realtimePlayer.stop()", after_resolution)
        self.assertRegex(
            self.view,
            r"(?s)\.fileImporter\(.*?realtimePlayer\.stop\(\).*?"
            r"Task \{ await store\.selectInput\(url\) \}",
        )
        self.assertRegex(
            self.view,
            r'(?s)Button\("入力を解除".*?realtimePlayer\.stop\(\).*?'
            r"store\.clearInput\(\)",
        )

    def test_runtime_master_variant_relative_urls_and_byte_range_materialization(self):
        requests = self.run_resolver_harness("master")
        paths = [urlsplit(path).path for _, path, _ in requests]
        self.assertIn("/hls/master.m3u8", paths)
        self.assertIn("/hls/high/index.m3u8", paths)
        self.assertNotIn("/hls/low/index.m3u8", paths)
        ranges = [value for _, path, value in requests if path.startswith("/hls/assets/")]
        self.assertEqual(ranges, ["bytes=0-3", "bytes=4-6", "bytes=7-8"])

    def test_runtime_nested_master_retains_inner_separate_audio_group(self):
        requests = self.run_resolver_harness("nested-master-audio")
        paths = [urlsplit(path).path for _, path, _ in requests]
        self.assertIn("/nested-audio/outer.m3u8", paths)
        self.assertIn("/nested-audio/inner/master.m3u8", paths)
        self.assertIn("/nested-audio/inner/video.m3u8", paths)

    def test_runtime_variant_failover_is_direct_ordered_and_context_preserving(self):
        requests = self.run_resolver_harness("variant-failover")
        paths = [urlsplit(path).path for _, path, _ in requests]
        self.assertLess(
            paths.index("/variant-switch/1080/video.m3u8"),
            paths.index("/variant-switch/720/index.m3u8"),
        )
        self.assertLess(
            paths.index("/variant-switch/720/index.m3u8"),
            paths.index("/variant-switch/480/index.m3u8"),
        )
        for path in [
            "/variant-switch/720/index.m3u8",
            "/variant-switch/480/index.m3u8",
        ]:
            methods = [
                method
                for method, request_path, _ in requests
                if urlsplit(request_path).path == path
            ]
            self.assertEqual(methods, ["GET"])

        self.assert_contracts(self.resolver, [
            "struct IPadHLSAlternativeVariant: Sendable, Equatable",
            "let masterURL: URL",
            "let playlistURL: URL",
            "let alternativeVariants: [IPadHLSAlternativeVariant]",
            "func resolveNextHLSVariant(",
            "submittedURL: source.submittedURL",
            "playbackURL: source.playbackURL",
            "resolutionPolicy: source.resolutionPolicy",
            "requestContext: source.requestContext",
            "nestedMetadata.retainingAlternativeVariants",
        ])
        failover = self.resolver.split("func resolveNextHLSVariant(", 1)[1].split(
            "/// Resolves static embed chains", 1
        )[0]
        self.assertIn("let resolved = try await resolveHLS(", failover)
        self.assertNotIn('method: "HEAD"', failover)

    def test_runtime_parent_audio_survives_nested_variant_failover(self):
        requests = self.run_resolver_harness("parent-audio-nested-failover")
        paths = [urlsplit(path).path for _, path, _ in requests]
        expected = [
            "/parent-audio-nested/inner/1080.m3u8",
            "/parent-audio-nested/inner/720.m3u8",
            "/parent-audio-nested/inner/480.m3u8",
            "/parent-audio-nested/inner/360.m3u8",
        ]
        self.assertEqual([path for path in paths if path in expected], expected)
        for path in expected[1:]:
            methods = [
                method
                for method, request_path, _ in requests
                if urlsplit(request_path).path == path
            ]
            self.assertEqual(methods, ["GET"])
        self.assert_contracts(self.resolver, [
            "hasAudioGroupMetadata",
            "inheritingAudioGroupIfMissing",
            "audioGroupID: retainedAudioGroupID",
            "audioRenditions: matchingAudio",
            "nestedMetadata.hasAudioGroupMetadata",
        ])

    def test_runtime_coalesces_identical_url_and_range_requests(self):
        requests = self.run_resolver_harness("shared-transport")
        paths = [urlsplit(path).path for _, path, _ in requests]
        self.assertEqual(paths.count("/shared/segment.ts"), 1)

    def test_runtime_cancelled_subscriber_does_not_cancel_shared_request(self):
        requests = self.run_resolver_harness("shared-cancellation")
        paths = [urlsplit(path).path for _, path, _ in requests]
        self.assertEqual(paths.count("/shared/cancel.ts"), 1)

    def test_runtime_retry_after_cooldown_is_shared_per_host(self):
        requests = self.run_resolver_harness("rate-limit-cooldown")
        paths = [urlsplit(path).path for _, path, _ in requests]
        self.assertEqual(paths.count("/rate-limit/first.ts"), 1)
        self.assertEqual(paths.count("/rate-limit/second.ts"), 1)

    def test_runtime_redirect_destination_cooldown_delays_direct_cdn_request(self):
        requests = self.run_resolver_harness("redirect-origin-cooldown")
        self.assertEqual(requests, [])

    def test_browser_relay_range_normalizer_synthesizes_complete_206_headers(self):
        requests = self.run_resolver_harness("browser-relay-range-normalizer")
        self.assertEqual(requests, [])

    def test_browser_relay_dispatch_gate_serializes_and_honors_429_cooldown(self):
        requests = self.run_resolver_harness("browser-relay-dispatch-gate")
        self.assertEqual(requests, [])

    def test_shared_transport_priority_and_permit_ownership_are_explicit(self):
        self.assert_contracts(self.resolver, [
            "case critical",
            "case speculative",
            "requestPriorities",
            "activeRequestPriorities",
            "higherPriorityIsWaiting",
            "func promote(",
            "containsEntry(entryToPromote)",
            "cancelAcquire(",
            "func recordOutcome(",
            "hasDistinctOutcomeOrigin",
            "rateLimitedRedirectRoute",
            "outcomeHostKey: Self.hostKey(for: entry.response?.url)",
            "var didAcquirePermit = false",
            "didAcquirePermit = true",
            "if didAcquirePermit",
            "requestID: entry.id",
            "priority: IPadSharedHTTPTransportOptions.Priority = .normal",
        ])

        coordinator = self.resolver.split(
            "private actor IPadHTTPOriginCoordinator", 1
        )[1].split("/// A process-wide, long-lived URLSession", 1)[0]
        self.assertIn("var congestionWindow = 1.0", coordinator)
        acquire = coordinator.split("func acquire(", 1)[1].split(
            "\n  func release(", 1
        )[0]
        self.assertIn(
            "state.requestPriorities[requestID] = effectivePriority",
            acquire,
        )
        self.assertIn(
            "pendingPriority.rawValue < effectivePriority.rawValue",
            acquire,
        )
        self.assertIn("!higherPriorityIsWaiting", acquire)
        self.assertIn(
            "state.rateLimitStrikes > 0 && effectivePriority == .speculative",
            acquire,
        )
        self.assertIn("!speculativeRecoveryBlocked", acquire)
        self.assertGreaterEqual(acquire.count("states[hostKey] = state"), 2)
        self.assertLess(
            acquire.rindex("states[hostKey] = state"),
            acquire.index("try await Task.sleep"),
        )

        outcome = coordinator.split("private func applyOutcome(", 1)[1].split(
            "\n  func promote(", 1
        )[0]
        self.assertIn("guard now >= state.cooldownUntil else { return }", outcome)
        self.assertIn("? max(4, window * 2)", outcome)
        self.assertIn(
            "state.rateLimitStrikes = max(0, state.rateLimitStrikes - 1)",
            outcome,
        )
        self.assertLess(
            outcome.index("if state.additiveSuccesses >= threshold"),
            outcome.index(
                "state.rateLimitStrikes = max(0, state.rateLimitStrikes - 1)"
            ),
        )

        settlement = self.resolver.split(
            "private static func releaseOriginPermit(", 1
        )[1].split("\n  #if MIOH_TESTING", 1)[0]
        self.assertIn("let rateLimitedRedirectRoute =", settlement)
        self.assertIn("hasDistinctOutcomeOrigin\n      && statusCode == 429", settlement)
        self.assertIn(
            "!hasDistinctOutcomeOrigin || rateLimitedRedirectRoute",
            settlement,
        )

    def test_rate_limit_runtime_probe_covers_route_priority_and_slow_aimd(self):
        probe = self.resolver.split(
            "static func redirectedOriginCooldownProbeForTesting()", 1
        )[1].split("\n  #endif", 1)[0]
        for contract in [
            "initialRouteElapsed",
            "destinationElapsed",
            "verifyCriticalPriorityAfterRateLimitForTesting()",
            "waitUntilQueuedForTesting(",
            "guard firstPriority == .critical",
            "verifySlowAIMDRecoveryForTesting()",
            "let expectedWindow = successIndex < 4 ? 1.0 : 2.0",
        ]:
            with self.subTest(contract=contract):
                self.assertIn(contract, probe)

    def test_runtime_html_discovery_prefers_hls_over_progressive_media(self):
        requests = self.run_resolver_harness("html")
        paths = [urlsplit(path).path for _, path, _ in requests]
        self.assertIn("/embed/page", paths)
        self.assertIn("/embed/stream/live.m3u8", paths)
        self.assertNotIn("/fallback.mp4", paths)

    def test_runtime_traverses_nested_static_players_and_prefers_hls(self):
        requests = self.run_resolver_harness("nested-pages")
        paths = [urlsplit(path).path for _, path, _ in requests]
        self.assertIn("/layers/root", paths)
        self.assertIn("/layers/one", paths)
        self.assertIn("/layers/two", paths)
        self.assertIn("/layers/three", paths)
        self.assertIn("/layers/final.m3u8", paths)
        self.assertNotIn("/layers/fallback.mp4", paths)
        self.assertNotIn("/layers/private-trap", paths)
        self.assertEqual(paths.count("/layers/root"), 2, "embed loop was fetched again")

    def test_runtime_stops_static_player_traversal_at_depth_limit(self):
        requests = self.run_resolver_harness("page-depth")
        paths = [urlsplit(path).path for _, path, _ in requests]
        self.assertIn("/depth/one", paths)
        self.assertIn("/depth/two", paths)
        self.assertIn("/depth/three", paths)
        self.assertNotIn("/depth/four", paths)
        self.assertNotIn("/depth/late.m3u8", paths)
        self.assertNotIn("/depth/fallback.mp4", paths)

    def test_runtime_enforces_cumulative_body_budget_across_page_layers(self):
        requests = self.run_resolver_harness("body-budget")
        paths = [urlsplit(path).path for _, path, _ in requests]
        self.assertEqual(paths.count("/body-budget/root"), 2)
        self.assertIn("/body-budget/one", paths)
        self.assertNotIn("/body-budget/two", paths)

    def test_runtime_rejects_drm_playlist(self):
        self.run_resolver_harness("drm")

    def test_runtime_allows_aes128_only_when_explicitly_enabled(self):
        self.run_resolver_harness("aes128")

    def test_runtime_tracks_hls_discontinuity_groups(self):
        requests = self.run_resolver_harness("discontinuity")
        self.assertIn(
            "/hls/discontinuity.m3u8",
            [urlsplit(path).path for _, path, _ in requests],
        )

    def test_runtime_rejects_invalid_durations_and_media_sequences(self):
        requests = self.run_resolver_harness("invalid-playlists")
        requested_paths = {urlsplit(path).path for _, path, _ in requests}
        self.assertTrue(
            {
                "/invalid/extinf-infinity.m3u8",
                "/invalid/extinf-zero.m3u8",
                "/invalid/extinf-overflow.m3u8",
                "/invalid/duplicate-media-sequence.m3u8",
                "/invalid/late-media-sequence.m3u8",
                "/invalid/media-sequence-overflow.m3u8",
            }.issubset(requested_paths)
        )

    def test_runtime_enforces_request_budget_before_fetching_variants(self):
        requests = self.run_resolver_harness("request-budget")
        paths = [urlsplit(path).path for _, path, _ in requests]
        self.assertEqual(paths.count("/hls/master.m3u8"), 2)
        self.assertNotIn("/hls/high/index.m3u8", paths)
        self.assertNotIn("/hls/low/index.m3u8", paths)

    def test_runtime_follows_redirect_and_uses_final_url_as_relative_base(self):
        requests = self.run_resolver_harness("redirect")
        paths = [urlsplit(path).path for _, path, _ in requests]
        self.assertIn("/go", paths)
        self.assertIn("/redirected/media.m3u8", paths)

    def test_runtime_enforces_redirect_limit(self):
        requests = self.run_resolver_harness("redirect-limit")
        redirect_requests = [
            path for _, path, _ in requests if path.startswith("/redirect-loop/")
        ]
        self.assertGreaterEqual(len(redirect_requests), 2)

    def test_runtime_rejects_cross_origin_redirect_to_local_address(self):
        requests = self.run_resolver_harness("redirect-private")
        paths = [urlsplit(path).path for _, path, _ in requests]
        self.assertGreaterEqual(paths.count("/redirect-private"), 1)
        self.assertNotIn("/layers/private-trap", paths)

    def test_runtime_cancellation_interrupts_an_active_request(self):
        started = time.monotonic()
        requests = self.run_resolver_harness("cancellation", timeout=4)
        self.assertLess(time.monotonic() - started, 2)
        self.assertTrue(any(urlsplit(path).path == "/slow" for _, path, _ in requests))


if __name__ == "__main__":
    unittest.main()
