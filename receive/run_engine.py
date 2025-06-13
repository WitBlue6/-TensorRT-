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
import os
import ctypes
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import time
import random
from typing import List, Optional, Dict ,Tuple

# Constants from yolov5_det_trt.py
CONF_THRESH = 0.5
IOU_THRESHOLD = 0.4
LEN_ALL_RESULT = 38001
LEN_ONE_RESULT = 38

# Load custom plugin library
ctypes.CDLL('./build/libmyplugins.so')

# COCO class names from yolov5_det_trt.py
categories = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"]

class YoLov5TRT:
    """
    Modified version from yolov5_det_trt.py with the same detection functionality
    but adapted to work within our jetson.py framework
    """
    def __init__(self, engine_file_path):
        # Create a Context on this device
        self.ctx = cuda.Device(0).make_context()
        self.stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        self.context = engine.create_execution_context()

        # Setup buffers
        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.bindings = []

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            self.bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)

        self.engine = engine
        self.batch_size = engine.max_batch_size

    def infer(self, image):
        """
        Single image inference adapted from yolov5_det_trt.py
        Returns:
            - Processed image with detections
            - Detection information dictionary
        """
        self.ctx.push()
        try:
            # Preprocess the image
            input_image, image_raw, origin_h, origin_w = self.preprocess_image(image)
            
            # Copy input image to host buffer
            np.copyto(self.host_inputs[0], input_image.ravel())
            
            # Transfer input data to the GPU
            cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
            
            # Run inference
            self.context.execute_async(batch_size=1, bindings=self.bindings, stream_handle=self.stream.handle)
            
            # Transfer predictions back from the GPU
            cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
            
            # Synchronize the stream
            self.stream.synchronize()
            
            # Get output
            output = self.host_outputs[0]
            
            # Post-process
            result_boxes, result_scores, result_classid = self.post_process(
                output[:LEN_ALL_RESULT], origin_h, origin_w
            )
            
            # Prepare detection info dictionary
            detection_info = {
                "object_count": len(result_boxes),
                "objects": [],
                "class_counts": {},
                "inference_time_ms": 0  # Will be set by detect_image
            }
            
            # Draw boxes and populate detection info
            for j in range(len(result_boxes)):
                box = result_boxes[j]
                score = result_scores[j]
                class_id = int(result_classid[j])
                class_name = categories[class_id]
                
                # Update class counts
                detection_info["class_counts"][class_name] = detection_info["class_counts"].get(class_name, 0) + 1
                
                # Add object info
                detection_info["objects"].append({
                    "class": class_name,
                    "confidence": float(score),
                    "position": {
                        "x1": int(box[0]),
                        "y1": int(box[1]),
                        "x2": int(box[2]),
                        "y2": int(box[3])
                    }
                })
                
                # Draw bounding box
                plot_one_box(
                    box,
                    image_raw,
                    label="{}:{:.2f}".format(class_name, score)
                )
            
            return image_raw, detection_info
            
        finally:
            self.ctx.pop()

    def preprocess_image(self, raw_bgr_image):
        """Same as in yolov5_det_trt.py"""
        image_raw = raw_bgr_image
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        
        # Calculate width and height and paddings
        r_w = self.input_w / w
        r_h = self.input_h / h
        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.input_h - th) / 2)
            ty2 = self.input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = self.input_h
            tx1 = int((self.input_w - tw) / 2)
            tx2 = self.input_w - tw - tx1
            ty1 = ty2 = 0
            
        # Resize and pad
        image = cv2.resize(image, (tw, th))
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, None, (128, 128, 128)
				)
        image = image.astype(np.float32)
        image /= 255.0
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w

    def xywh2xyxy(self, origin_h, origin_w, x):
        """Same as in yolov5_det_trt.py"""
        y = np.zeros_like(x)
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h
        return y

    def post_process(self, output, origin_h, origin_w):
        """Same as in yolov5_det_trt.py"""
        num = int(output[0])
        pred = np.reshape(output[1:], (-1, LEN_ONE_RESULT))[:num, :]
        pred = pred[:, :6]
        
        # Do nms
        boxes = self.non_max_suppression(pred, origin_h, origin_w, conf_thres=CONF_THRESH, nms_thres=IOU_THRESHOLD)
        result_boxes = boxes[:, :4] if len(boxes) else np.array([])
        result_scores = boxes[:, 4] if len(boxes) else np.array([])
        result_classid = boxes[:, 5] if len(boxes) else np.array([])
        return result_boxes, result_scores, result_classid

    def non_max_suppression(self, prediction, origin_h, origin_w, conf_thres=0.5, nms_thres=0.4):
        """Same as in yolov5_det_trt.py"""
        boxes = prediction[prediction[:, 4] >= conf_thres]
        boxes[:, :4] = self.xywh2xyxy(origin_h, origin_w, boxes[:, :4])
        boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w -1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w -1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h -1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h -1)
        
        confs = boxes[:, 4]
        boxes = boxes[np.argsort(-confs)]
        
        keep_boxes = []
        while boxes.shape[0]:
            large_overlap = self.bbox_iou(np.expand_dims(boxes[0, :4], 0), boxes[:, :4]) > nms_thres
            label_match = boxes[0, -1] == boxes[:, -1]
            invalid = large_overlap & label_match
            keep_boxes += [boxes[0]]
            boxes = boxes[~invalid]
            
        boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
        return boxes

    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        """Same as in yolov5_det_trt.py"""
        if not x1y1x2y2:
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        
        inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * \
                     np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
        
        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
        return iou

    def __del__(self):
        """Cleanup CUDA context"""
        if hasattr(self, 'ctx') and self.ctx:
            try:
                self.ctx.pop()
            except:
                pass

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    Same as in yolov5_det_trt.py
    description: Plots one bounding box on image img
    """
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )

def detect_image(img: np.ndarray, yolov5: YoLov5TRT) -> Tuple[np.ndarray, Dict]:
    """
    Simplified version that uses the YoLov5TRT class for detection
    """
    start_time = time.time()
    processed_img, detection_info = yolov5.infer(img)
    detection_info["inference_time_ms"] = round((time.time() - start_time) * 1000, 2)
    return processed_img, detection_info

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
    def __init__(self, my_ip, my_port, peer_uri, engine_path):
        self.my_ip = my_ip
        self.my_port = my_port
        self.peer_uri = peer_uri
        self.latest_image = None
        self.frame_rate = 3
        self.compression_quality = 95
        self._running = True
        self._display_thread = None
        self._loop = None
        self.yolov5 =YoLov5TRT('./build/yolov5s-best.engine')
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
                        img_process, detection_info = detect_image(img, self.yolov5)
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
        "10.171.103.229",  # jetson
        8765,
        "ws://10.194.128.156:8765",  # 上位机
        './build/yolov5s-best.engine'
    )
    transfer.run()