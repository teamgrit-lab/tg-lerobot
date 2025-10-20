import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
import asyncio, websockets, time

url = ""
Gst.init(None)

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