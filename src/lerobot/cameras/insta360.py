import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
import asyncio, websockets, time
import os
from pathlib import Path

def _find_project_root() -> Path:
        here = Path(__file__).resolve()
        for p in here.parents:
                if (p / "pyproject.toml").exists():
                        return p
        return here.parent


def _load_host_port_from_common_yaml() -> tuple[str, int]:
        host = "localhost"
        port = 8765
        try:
                root = _find_project_root()
                cfg = root / "configs" / "ws_common.yaml"
                if cfg.exists():
                        with cfg.open("r", encoding="utf-8") as f:
                                for line in f:
                                        line = line.strip()
                                        if not line or line.startswith("#"):
                                                continue
                                        if line.startswith("host:"):
                                                host = line.split(":", 1)[1].strip()
                                        elif line.startswith("port:"):
                                                port_str = line.split(":", 1)[1].strip()
                                                try:
                                                        port = int(port_str)
                                                except ValueError:
                                                        pass
        except Exception:
                pass
        return host, port


def _parse_hint_env(env_var: str) -> tuple[str, ...] | None:
        raw = os.getenv(env_var, "").strip()
        if not raw:
                return None
        hints = [part.strip().lower() for part in raw.split(",") if part.strip()]
        return tuple(hints) if hints else None


def _matches_hints(text: str, hints: tuple[str, ...] | None) -> bool:
        if not hints:
                return True
        return any(hint in text for hint in hints)


def _iter_v4l2_by_id() -> list[Path]:
        by_id_dir = Path("/dev/v4l/by-id")
        if not by_id_dir.exists():
                return []
        return sorted(by_id_dir.iterdir())


def _pick_v4l2_by_id(name_hints: tuple[str, ...] | None) -> str | None:
        candidates: list[Path] = []
        for entry in _iter_v4l2_by_id():
                entry_name = entry.name.lower()
                if not _matches_hints(entry_name, name_hints):
                        continue
                candidates.append(entry)
        if not candidates:
                return None
        for entry in candidates:
                if "index0" in entry.name:
                        return str(entry.resolve(strict=False))
        return str(candidates[0].resolve(strict=False))


def _pick_v4l2_by_sysfs(name_hints: tuple[str, ...]) -> str | None:
        sys_dir = Path("/sys/class/video4linux")
        if not sys_dir.exists():
                return None
        for video in sorted(sys_dir.glob("video*")):
                name_path = video / "name"
                if not name_path.exists():
                        continue
                name = name_path.read_text(encoding="utf-8", errors="ignore").strip().lower()
                if _matches_hints(name, name_hints):
                        return f"/dev/{video.name}"
        return None


def _resolve_v4l2_device(
        default_device: str,
        device_env: str,
        name_hints: tuple[str, ...] | None = None,
) -> str:
        env_device = os.getenv(device_env, "").strip()
        if env_device:
                return env_device
        device = _pick_v4l2_by_id(name_hints)
        if device:
                return device
        if name_hints:
                device = _pick_v4l2_by_sysfs(name_hints)
                if device:
                        return device
        return default_device


def _wait_for_device(device_path: str, timeout_s: float = 10.0) -> None:
        if Path(device_path).exists():
                return
        deadline = time.time() + timeout_s
        while time.time() < deadline:
                if Path(device_path).exists():
                        return
                time.sleep(0.2)
        print(f"warning: device not found after {timeout_s:.1f}s: {device_path}")


_host, _port = _load_host_port_from_common_yaml()
url = f"ws://{_host}:{_port}/pang/ws/pub?channel=instant&name=test&track=insta360&mode=single"
_INSTA_HINTS_ENV = "LEROBOT_INSTA360_HINTS"
_INSTA_DEVICE_ENV = "LEROBOT_INSTA360_DEVICE"
_insta_hints = _parse_hint_env(_INSTA_HINTS_ENV) or ("insta360", "insta", "link")
DEVICE_PATH = _resolve_v4l2_device(
        default_device="/dev/video12",
        device_env=_INSTA_DEVICE_ENV,
        name_hints=_insta_hints,
)
_wait_for_device(DEVICE_PATH)
print(f"insta360 device: {DEVICE_PATH}")
Gst.init(None)

PIPELINE_DESC = (
    f'v4l2src device="{DEVICE_PATH}" ! '
    "image/jpeg, width=2880, height=1440, framerate=30/1 ! "
    "jpegparse ! "
    "jpegdec ! "
    "videoconvert ! videoscale ! video/x-raw, width=1440, height=720 ! videoconvert ! "
#    "video/x-raw, format=I420 ! "
#    "openh264enc bitrate=4000000 gop-size=30 ! "
#    "video/x-h264, stream-format=byte-stream ! "
    "nvvidconv ! "
    "nvv4l2h264enc bitrate=4000000 idrinterval=30 iframeinterval=30 insert-sps-pps=true ! "
    "video/x-h264, stream-format=byte-stream, alignment=au ! "
    "queue leaky=downstream ! "
    "appsink name=sink drop=true max-buffers=1 emit-signals=true sync=false"
)

def _start_pipeline() -> tuple[Gst.Pipeline, Gst.Element]:
    pipeline = Gst.parse_launch(PIPELINE_DESC)
    sink = pipeline.get_by_name("sink")
    pipeline.set_state(Gst.State.PLAYING)
    pipeline.get_state(2 * Gst.SECOND)
    return pipeline, sink

pipeline, sink = _start_pipeline()

def restart():
        global pipeline, sink
        if pipeline is not None:
                pipeline.set_state(Gst.State.NULL)
                pipeline.get_state(2 * Gst.SECOND)
        pipeline, sink = _start_pipeline()

async def recv(ws):
        print("recv")
        while True:
                data = await ws.recv()
                print(data[0:11])
                await asyncio.sleep(0.01)

async def send(ws):
        print("send")
        now = time.time()
        count = 0
        await ws.send('video/h264;width=1440;height=720;framerate=30;codecs=avc1.42002A')
        while True:
                try:
                        sample = sink.emit("pull-sample")
                        if time.time() - now > 1:
                                now = time.time()
                                print(count, "fps")
                                if count == 0:
                                        restart()
                                count = 0
                        if sample:
                                buf = sample.get_buffer()
                                data = buf.extract_dup(0, buf.get_size())
                                await ws.send(data)
                                count += 1
                        await asyncio.sleep(0.01)
                except:
                        await asyncio.sleep(0.1)

async def main():
        while True:
                try:
                        async with websockets.connect(url, ping_timeout=None) as ws:
                                t1 = asyncio.create_task(recv(ws))
                                t2 = asyncio.create_task(send(ws))
                                try:
                                        await asyncio.gather(t1, t2)
                                finally:
                                        t1.cancel()
                                        t2.cancel()
                                        await asyncio.gather(t1, t2, return_exceptions=True)
                except Exception as exc:
                        print("ws connect failed, retrying", exc)
                        await asyncio.sleep(1)

asyncio.run(main())