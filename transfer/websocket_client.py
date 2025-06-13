import asyncio
import sys
import websockets
import cv2
import numpy as np
import json
from PyQt5.QtCore import QThread, pyqtSignal

CHUNK_SIZE = 1024 * 1024


class RealTimeImageTransfer(QThread):
    image_received = pyqtSignal(np.ndarray)
    detection_info_received = pyqtSignal(dict)  # 新增信号用于传输检测信息
    status_update = pyqtSignal(str)
    connection_status = pyqtSignal(bool)

    def __init__(self, my_ip, my_port, peer_uri):
        super().__init__()
        self.my_ip = my_ip
        self.my_port = my_port
        self.peer_uri = peer_uri
        self.latest_image = None
        self.frame_rate = 20
        self.compression_quality = 95
        self._running = True
        self._display_thread = None
        self._loop = None
        self.frame = None
        self.send = False
        self.logger = print
        self.receive_buffer = bytearray()
        self.expecting_image = True  # 标记当前期望接收的是图像还是信息

    async def _image_sender(self, websocket):
        """发送图像"""
        while self._running:
            try:
                if self.frame is not None and self.send:
                    _, buffer = cv2.imencode('.jpg', self.frame)  # 图像编码
                    buffer_bytes = buffer.tobytes()
                    buffer_length = len(buffer_bytes)
                    offset = 0
                    while offset < buffer_length:  # 分批次传输图像
                        end = min(offset + CHUNK_SIZE, buffer_length)
                        await websocket.send(buffer_bytes[offset:end])
                        offset = end
                    self.send = False
                await asyncio.sleep(1 / self.frame_rate)  # 等待传输时间
            except websockets.exceptions.ConnectionClosed:
                self.logger("发送时连接关闭")
                break
            except Exception as e:
                self.logger(f"发送错误: {e}")
                break

    async def _image_receiver(self, websocket):
        """接收处理后的图像和检测信息"""
        self.receive_buffer = bytearray()
        self.expecting_image = True

        while self._running:
            try:
                data = await websocket.recv()

                if self.expecting_image:
                    # 正在接收图像数据
                    if isinstance(data, bytes):
                        self.receive_buffer.extend(data)
                        if len(data) < CHUNK_SIZE:
                            # 图像接收完成
                            img = cv2.imdecode(np.  array(self.receive_buffer), cv2.IMREAD_COLOR)
                            self.image_received.emit(img)
                            self.receive_buffer = bytearray()
                            self.expecting_image = False  # 下一帧期待接收信息
                else:
                    # 正在接收检测信息
                    if isinstance(data, bytes):
                        try:
                            info = json.loads(data.decode('utf-8'))
                            self.detection_info_received.emit(info)
                        except Exception as e:
                            self.logger(f"解析检测信息错误: {e}")
                    self.expecting_image = True  # 下一帧期待接收图像

            except websockets.exceptions.ConnectionClosedOK:
                self.status_update.emit("连接正常关闭")
                break
            except Exception as e:
                self.status_update.emit(f"接收错误: {e}")
                break

    async def _handler(self, websocket):
        """处理单个连接会话"""
        try:
            await asyncio.gather(
                self._image_sender(websocket),
                self._image_receiver(websocket)
            )
        except Exception as e:
            self.logger(f"处理器错误: {e}")
        finally:
            await websocket.close()

    def start_server(self):
        """启动WebSocket服务器"""

        async def _run_server():
            async with websockets.serve(
                    self._handler,
                    self.my_ip,
                    self.my_port,
                    ping_interval=30,
                    ping_timeout=60,
                    close_timeout=1,
                    max_size=10 * 1024 * 1024,
            ):
                self.logger(f"服务器已启动 ws://{self.my_ip}:{self.my_port}")
                await asyncio.Future()  # 永久运行

        def run_in_thread():
            asyncio.run(_run_server())

        import threading
        threading.Thread(target=run_in_thread, daemon=True).start()

    async def _connect_with_retry(self, max_retries=5, delay=3):
        for attempt in range(max_retries):
            try:
                async with websockets.connect(
                        self.peer_uri,
                        ping_interval=30,
                        ping_timeout=60,
                        close_timeout=1
                ) as websocket:
                    #self.logger("连接成功!")
                    self.connection_status.emit(True)  # 连接成功时发射信号
                    await self._handler(websocket)
                    return True
            except Exception as e:
                #self.logger(f"尝试 {attempt+1}/{max_retries} 失败: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
        asyncio.sleep(delay)
        #self.logger("所有连接尝试均失败，请退出程序。")
        self.connection_status.emit(False)  # 连接失败时发射信号
        sys.exit(1)  # 超过5次连接失败，自动退出程序

    def _start_server_and_connect(self):
        """在子线程运行的服务器和连接逻辑"""
        self.start_server()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._connect_with_retry())

    def run(self):
        # 在子线程启动服务器
        import threading
        threading.Thread(target=self._start_server_and_connect, daemon=True).start()

    def stop(self):
        """停止运行"""
        self._running = False
        if self._loop:
            self._loop.stop()
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        if self._display_thread and self._display_thread.is_alive():
            self._display_thread.join()