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
url = f"ws://{_host}:{_port}/pang/ws/pub?channel=instant&name=test&track=hand_camera&mode=single"
Gst.init(None)

PIPELINE_DESC = (
        "v4l2src device=/dev/video1 ! "
        "image/jpeg, width=1280, height=720, framerate=30/1 ! "
        "jpegparse ! "
        "jpegdec ! "
#        "x264enc bitrate=4096 speed-preset=1 key-int-max=30 tune=zerolatency ! "
        "videoconvert ! nvvidconv ! "
        "nvv4l2h264enc bitrate=4000000 idrinterval=30 iframeinterval=30 insert-sps-pps=true ! "
        "h264parse ! "
        "video/x-h264, alignment=au, stream-format=byte-stream ! "
        "queue leaky=2 ! appsink drop=true sync=false name=sink max-buffers=3 emit-signals=true"
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
        await ws.send('video/h264;width=1280;height=720;framerate=30;codecs=avc1.42002A')
        while True:
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
