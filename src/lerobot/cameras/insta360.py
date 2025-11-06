import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
import asyncio, websockets, time
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


_host, _port = _load_host_port_from_common_yaml()
url = f"ws://{_host}:{_port}/pang/ws/pub?channel=instant&name=test&track=insta360&mode=single"
Gst.init(None)

global pipeline
pipeline = Gst.parse_launch(
    "v4l2src device=/dev/video12 ! "
    "image/jpeg, width=2880, height=1440, framerate=30/1 ! "
    "jpegparse ! "
    "jpegdec ! "
    "videoconvert ! videoscale ! video/x-raw, width=1440, height=720 ! videoconvert ! "
    "video/x-raw, format=I420 ! "
    "openh264enc bitrate=4000000 gop-size=30 ! "
    "video/x-h264, stream-format=byte-stream ! "
    "queue leaky=downstream ! "
    "appsink name=sink drop=true max-buffers=1 emit-signals=true sync=false"
)
sink = pipeline.get_by_name('sink')
pipeline.set_state(Gst.State.PLAYING)
time.sleep(1)

def restart():
        global pipeline
        pipeline.set_state(Gst.State.NULL)
        time.sleep(1)
        pipeline.set_state(Gst.State.PLAYING)

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
        async with websockets.connect(url, ping_timeout=None) as ws:
                t1 = asyncio.create_task(recv(ws))
                t2 = asyncio.create_task(send(ws))
                await asyncio.gather(t1, t2)

asyncio.run(main())