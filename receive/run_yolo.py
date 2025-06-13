import asyncio
import websockets
import cv2
import numpy as np
import json
from threading import Thread
import logging
from PyQt5.QtWidgets import QApplication, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
import sys
#from test import process_image_data
from yolo import detect_image

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WebSocketTransfer")
CHUNK_SIZE = 1024 * 1024


class ImageDisplay:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.label = QLabel()
        self.label.setWindowTitle("Received Image")
        self.label.resize(800, 600)
        self.label.show()
        self.timer = QTimer()

    def start(self, update_callback):
        self.timer.timeout.connect(update_callback)
        self.timer.start(30)
        sys.exit(self.app.exec_())


class RealTimeImageTransfer:
    def __init__(self, my_ip, my_port, peer_uri):
        self.my_ip = my_ip
        self.my_port = my_port
        self.peer_uri = peer_uri
        self.latest_image = None
        self.frame_rate = 3
        self.compression_quality = 95
        self._running = True
        self._display_thread = None
        self._loop = None

    async def _image_receiver(self, websocket):
        received_data = bytearray()
        while self._running:
            try:
                data = await websocket.recv()

                if isinstance(data, bytes):
                    received_data.extend(data)
                    if len(data) < CHUNK_SIZE:
                        # 接收完整图像
                        nparr = np.frombuffer(received_data, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        logger.info(f"收到图像: {img.shape}")

                        # 处理图像并获取结果和检测信息
                        img_process, detection_info = detect_image(img, weights="best.pt", device="0")
                        self.latest_image = img_process

                        # 先发送处理后的图像
                        _, buffer = cv2.imencode('.jpg', img_process)
                        buffer_bytes = buffer.tobytes()
                        buffer_length = len(buffer_bytes)
                        offset = 0
                        while offset < buffer_length:
                            end = min(offset + CHUNK_SIZE, buffer_length)
                            await websocket.send(buffer_bytes[offset:end])
                            offset = end

                        # 然后发送检测信息
                        info_json = json.dumps(detection_info)
                        await websocket.send(info_json.encode('utf-8'))

                        logger.info("已返回处理后的图像和检测信息")
                        received_data = bytearray()

            except websockets.exceptions.ConnectionClosed:
                logger.warning("接收时连接关闭")
                break
            except Exception as e:
                logger.error(f"接收错误: {e}")
                break

    async def _handler(self, websocket):
        try:
            await self._image_receiver(websocket)
        except Exception as e:
            logger.error(f"处理器错误: {e}")
        finally:
            await websocket.close()

    def start_server(self):
        async def _run_server():
            async with websockets.serve(
                    self._handler,
                    self.my_ip,
                    self.my_port
            ):
                logger.info(f"服务器已启动 ws://{self.my_ip}:{self.my_port}")
                await asyncio.Future()

        def run_in_thread():
            asyncio.run(_run_server())

        Thread(target=run_in_thread, daemon=True).start()

    async def _connect_with_retry(self, max_retries=50, delay=3):
        for attempt in range(max_retries):
            try:
                async with websockets.connect(
                        self.peer_uri,
                        ping_interval=30,
                        ping_timeout=60,
                        close_timeout=1
                ) as websocket:
                    logger.info("连接成功!")
                    await self._handler(websocket)
                    return True
            except Exception as e:
                logger.warning(f"尝试 {attempt + 1}/{max_retries} 失败: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
        return False

    def _start_server_and_connect(self):
        self.start_server()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._connect_with_retry())

    def run(self):
        display = ImageDisplay()

        def update_image():
            if self.latest_image is not None:
                img_rgb = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2RGB)
                h, w, ch = img_rgb.shape
                q_img = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
                display.label.setPixmap(QPixmap.fromImage(q_img))

        Thread(target=self._start_server_and_connect, daemon=True).start()
        display.start(update_image)


if __name__ == "__main__":
    transfer = RealTimeImageTransfer(
        "10.178.170.93",
        8765,
        "ws://10.194.139.132:8765"
    )
    transfer.run()
