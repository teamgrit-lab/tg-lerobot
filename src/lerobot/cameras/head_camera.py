import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
import asyncio, websockets, time

url = ""
Gst.init(None)

pipeline = Gst.parse_launch(
        "v4l2src device=/dev/video1 ! "
        "image/jpeg, width=1280, height=720, framerate=30/1 ! "
        "jpegparse ! "
        "jpegdec ! "
        "videoconvert ! "
        "x264enc bitrate=4096 speed-preset=1 key-int-max=30 tune=zerolatency ! "
        "h264parse ! "
        "video/x-h264, alignment=au, stream-format=byte-stream ! "
        "queue leaky=2 ! appsink drop=true sync=false name=sink max-buffers=3 emit-signals=true"
)
sink = pipeline.get_by_name('sink')
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
        await ws.send('video/h264;width=1280;height=720;framerate=30;codecs=avc1.42002A')
        while True:
                sample = sink.emit("pull-sample")
                if time.time() - now > 1:
                        now = time.time()
                        print(count, "fps")
                        count = 0
                if sample:
                        buf = sample.get_buffer()
                        data = buf.extract_dup(0, buf.get_size())
                        await ws.send(data)
                        count += 1
                await asyncio.sleep(0.01)

async def main():
        async with websockets.connect(url, ping_timeout=None) as ws:
                t1 = asyncio.create_task(recv(ws))
                t2 = asyncio.create_task(send(ws))
                await asyncio.gather(t1, t2)

asyncio.run(main())