import sys
import os
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QFileDialog, QLabel, QTextEdit, QSplitter,
                             QMessageBox, QScrollArea, QSizePolicy, QSlider, QLineEdit)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage,QIntValidator
from websocket_client import RealTimeImageTransfer
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pathlib import Path
import pyttsx3

class AgentWorker(QThread):
    result_ready = pyqtSignal(str)
    history_ready = pyqtSignal(object)

    def __init__(self, agent, prompt, history):
        super().__init__()
        self.agent = agent
        self.prompt = prompt
        self.history = history

    def run(self):
        resp = self.agent.run_sync(self.prompt, message_history=self.history)
        self.result_ready.emit(resp.output)
        self.history_ready.emit(list(resp.all_messages()))

class SpeechWorker(QThread):
    finished = pyqtSignal()

    def __init__(self, text):
        super().__init__()
        self.text = text
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 200)
        self.engine.setProperty('volume', 1.0)

    def run(self):
        self.engine.say(self.text)
        self.engine.runAndWait()
        self.finished.emit()

class DetectionApp(QMainWindow):
    def __init__(self, my_ip, my_port, server_uri):
        super().__init__()
        # 本地 IP 和端口
        self.my_ip = my_ip
        self.my_port = my_port
        # 服务器的 WebSocket URI
        self.server_uri = server_uri
        self.setWindowTitle("智能远程图像检测系统")
        self.setGeometry(100, 100, 1500, 1200)

        # 初始化变量
        self.current_image_path = None
        self.current_image = None
        self.result_image = None
        self.image_files = []
        self.current_index = 0

        # 初始化 WebSocket 客户端
        self.ws_client = RealTimeImageTransfer(my_ip, my_port, server_uri)
        self.ws_client.run()
        # 连接信号与槽
        self.ws_client.image_received.connect(self.handle_result_image)
        self.ws_client.status_update.connect(self.update_status)
        self.ws_client.connection_status.connect(self.update_connection_status)  # 新增连接状态信号槽
        self.ws_client.logger = self.update_status
        self.ws_client.detection_info_received.connect(self.handle_detection_info)
        # 初始化 UI
        self.initUI()
        # 缩放因子
        self.zoom_factor = 0.5

        # 自动播放定时器
        self.autoplay_timer = QTimer(self)
        self.autoplay_timer.timeout.connect(self.autoplay_next_image)
        self.is_autoplaying = False
        self.display_time = 200  # 默认显示时间为 10 秒

    def initUI(self):
        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # 创建主布局
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        # 创建控制面板
        control_panel = QWidget()
        control_layout = QHBoxLayout()
        control_panel.setLayout(control_layout)

        # 控制按钮
        self.btn_open_image = QPushButton("打开单张图片")
        self.btn_open_image.clicked.connect(self.open_image)

        self.btn_open_folder = QPushButton("打开文件夹")
        self.btn_open_folder.clicked.connect(self.open_folder)

        self.btn_prev = QPushButton("上一张")
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_prev.setEnabled(False)

        self.btn_next = QPushButton("下一张")
        self.btn_next.clicked.connect(self.next_image)
        self.btn_next.setEnabled(False)

        self.btn_detect = QPushButton("开始检测")
        self.btn_detect.clicked.connect(self.send_for_detection)
        self.btn_detect.setEnabled(False)

        # 自动播放按钮
        self.btn_autoplay = QPushButton("自动播放")
        self.btn_autoplay.clicked.connect(self.toggle_autoplay)
        self.btn_autoplay.setEnabled(False)

        # 初始化显示时间
        self.display_time = 200  # 默认显示时间为 10 秒

        # 显示时间输入框
        self.display_time_input = QLineEdit()
        self.display_time_input.setText(str(self.display_time))
        self.display_time_input.setValidator(QIntValidator(1, 10000))  # 限制输入范围为 1 - 60 秒
        self.display_time_input.textChanged.connect(self.update_display_time)

        # 连接状态指示器
        self.connection_indicator = QLabel()
        self.connection_indicator.setFixedSize(20, 20)
        self.update_connection_status(False)

        # 缩放滑块
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(10)
        self.zoom_slider.setMaximum(200)
        self.zoom_slider.setValue(50)
        self.zoom_slider.setTickInterval(10)
        self.zoom_slider.setTickPosition(QSlider.TicksBelow)
        self.zoom_slider.valueChanged.connect(self.update_zoom)

        control_layout.addWidget(self.connection_indicator)
        control_layout.addWidget(self.btn_open_image)
        control_layout.addWidget(self.btn_open_folder)
        control_layout.addWidget(self.btn_prev)
        control_layout.addWidget(self.btn_next)
        control_layout.addWidget(self.btn_autoplay)
        control_layout.addWidget(QLabel("显示时间 (毫秒):"))
        control_layout.addWidget(self.display_time_input)
        #control_layout.addStretch()
        control_layout.addWidget(self.btn_detect)
        control_layout.addWidget(QLabel("缩放:"))
        control_layout.addWidget(self.zoom_slider)
        # 图像显示区域
        splitter = QSplitter(Qt.Horizontal)

        # 原始图像显示区域
        self.original_scroll = QScrollArea()
        self.original_label = QLabel()
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setText("原始图像")
        self.original_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.original_scroll.setWidget(self.original_label)
        self.original_scroll.setWidgetResizable(True)

        # 检测结果显示区域
        self.result_scroll = QScrollArea()
        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setText("检测结果")
        self.result_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.result_scroll.setWidget(self.result_label)
        self.result_scroll.setWidgetResizable(True)

        splitter.addWidget(self.original_scroll)
        splitter.addWidget(self.result_scroll)
        splitter.setSizes([700, 700])

        # 状态信息显示区域
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)

        # 用户自然语言输入区域
        user_input_panel = QWidget()
        user_input_layout = QHBoxLayout()
        user_input_panel.setLayout(user_input_layout)

        self.user_input_line = QLineEdit()
        self.user_input_line.setPlaceholderText("请输入指令或问题，例如：‘发送目录中第一张图片’")
        self.btn_send_input = QPushButton("发送")
        self.btn_send_input.clicked.connect(self.send_user_input)

        user_input_layout.addWidget(QLabel("用户指令:"))
        user_input_layout.addWidget(self.user_input_line)
        user_input_layout.addWidget(self.btn_send_input)


        main_layout.addWidget(control_panel)
        main_layout.addWidget(splitter, stretch=1)
        main_layout.addWidget(self.info_text)
        main_layout.addWidget(user_input_panel)

        main_layout.setStretch(0, 0)
        main_layout.setStretch(1, 2)
        main_layout.setStretch(2, 1)

        # Agent初始化
        load_dotenv()
        # 配置 OpenAIModel
        model = OpenAIModel(
            "gpt-4o-mini",
            provider=OpenAIProvider(
                api_key=os.getenv('OPENAI_API_KEY'), 
                base_url=os.getenv('OPENAI_BASE_URL')
            )
        )
        self.agent = Agent(
            model,
            system_prompt="你是一个图像检测助手，你的任务是根据用户的指令和图像内容，向服务器发送要处理的图片。",
            tools = [self.user_read_file, self.user_list_file, self.user_load_image, self.update_display_time],
        )
        self.history = [] # 对话历史
        self.update_status('\n[Agent响应] 你好，我是你的图像检测助手，有什么可以帮你的吗？')
        

    def update_connection_status(self, connected):
        """更新连接状态指示器"""
        if connected:
            self.connection_indicator.setStyleSheet("background-color: green; border-radius: 10px;")
            self.connection_indicator.setToolTip("Connected to remote system")
            self.btn_open_image.setEnabled(True)
            self.btn_open_folder.setEnabled(True)
            self.btn_detect.setEnabled(True)
        else:
            self.connection_indicator.setStyleSheet("background-color: red; border-radius: 10px;")
            self.connection_indicator.setToolTip("Disconnected from remote system")
            self.btn_open_image.setEnabled(True)
            self.btn_open_folder.setEnabled(True)
            self.btn_detect.setEnabled(True)

    def open_image(self):
        """打开单张图片文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp)")

        if file_path:
            self.image_files = [file_path]
            self.current_index = 0
            self.load_current_image()
            self.btn_autoplay.setEnabled(False)

    def open_folder(self):
        """打开包含图片的文件夹"""
        folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹")

        if folder_path:
            self.image_files = [
                os.path.join(folder_path, f) for f in os.listdir(folder_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            ]

            if self.image_files:
                self.current_index = 0
                self.load_current_image()
                self.btn_autoplay.setEnabled(True)
            else:
                QMessageBox.warning(self, "警告", "文件夹中没有找到图片文件!")

    def load_current_image(self):
        """加载当前图片"""
        if 0 <= self.current_index < len(self.image_files):
            self.current_image_path = self.image_files[self.current_index]
            self.current_image = cv2.imread(self.current_image_path)

            if self.current_image is not None:
                self.display_image(self.current_image, self.original_label)
                self.result_image = None
                self.display_image(None, self.result_label)
                self.btn_detect.setEnabled(True)
                self.update_navigation_buttons()
                #self.info_text.clear()
            else:
                QMessageBox.warning(self, "错误", "无法加载图片文件!")
    def user_read_file(self, file_name: str) -> str:
        """读取本地存在的图片文件"""
        base_path = Path('./datas')
        file_path = base_path / file_name
        if not os.path.exists(file_path):
            return f"文件 {file_path} 不存在"
        with open(file_path, "r", encoding="utf-8") as f:
            print(f'Read file {file_path}')
            return f.read()
        
    def user_list_file(self) -> list[str]:
        '''查看目录下有哪些可用的图片'''
        base_path = Path('./datas')
        print('Check file list')
        file_list = []
        for item in base_path.rglob('*'):
            if item.is_file():
                file_list.append(str(item.relative_to(base_path).as_posix()))
        return file_list
    
    def user_load_image(self, file_name:str):
        """加载当前的图片并发送到服务器"""
        #self.btn_autoplay.setEnabled(False)
        base_path = Path('./datas')
        file_path = base_path / file_name
        if not os.path.exists(file_path):
            return f"文件 {file_path} 不存在"
        self.current_image_path = file_path
        self.current_image = cv2.imread(self.current_image_path)
        # 显示读取的图片
        if self.current_image is not None:
            self.display_image(self.current_image, self.original_label)
            self.result_image = None
            self.display_image(None, self.result_label)
            self.btn_detect.setEnabled(True)
            self.update_navigation_buttons()
            self.send_for_detection()
        

    def prev_image(self):
        """显示上一张图片"""
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()

    def next_image(self):
        """显示下一张图片"""
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_current_image()

    def update_navigation_buttons(self):
        """更新导航按钮的可用状态"""
        self.btn_prev.setEnabled(len(self.image_files) > 1 and self.current_index > 0)
        self.btn_next.setEnabled(len(self.image_files) > 1 and self.current_index < len(self.image_files) - 1)

    def send_for_detection(self):
        """发送当前图片进行远程检测"""
        if self.current_image is None:
            return

        # 处理过程中禁用按钮
        self.btn_detect.setEnabled(False)
        self.btn_open_image.setEnabled(False)
        self.btn_open_folder.setEnabled(False)

        # 显示处理状态
        self.result_label.setText("检测中...")
        self.update_status("正在发送图像到远程系统...")

        # 发送图片
        self.ws_client.frame = self.current_image
        self.ws_client.send = True

    def handle_result_image(self, image):
        """处理接收到的检测结果图片"""
        self.result_image = image
        self.display_image(self.result_image, self.result_label)

        # 重新启用按钮
        self.btn_detect.setEnabled(True)
        self.btn_open_image.setEnabled(True)
        self.btn_open_folder.setEnabled(True)

        self.update_status("检测完成")

        if self.is_autoplaying:
            self.autoplay_timer.start(self.display_time)

    def handle_detection_info(self, info):
        """处理接收到的检测信息"""
        self.info_text.append("\n=== 检测结果 ===")

        # 显示详细信息
        self.info_text.append("\n详细检测结果:")
        for obj in info["objects"]:
            self.info_text.append(
                f"- {obj['class']} (置信度: {obj['confidence']:.2f}), "
                f"位置: ({obj['position']['x1']}, {obj['position']['y1']}) "
                f"到 ({obj['position']['x2']}, {obj['position']['y2']})"
            )

        # 显示统计信息
        self.info_text.append(f"检测到 {info['object_count']} 个目标:")
        for class_name, count in info["class_counts"].items():
            self.info_text.append(f"  {class_name}: {count}个")

        self.info_text.append(f"检测耗时:{info['inference_time_ms']}毫秒")
        # 滚动到最新信息
        self.info_text.verticalScrollBar().setValue(
            self.info_text.verticalScrollBar().maximum()
        )

    def update_status(self, message):
        """更新状态信息"""
        self.info_text.append(message)

    def display_image(self, image, label):
        """在指定标签上显示图片"""
        if image is None:
            label.clear()
            label.setText("暂无图像")
            return

        h, w, ch = image.shape
        bytes_per_line = ch * w
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(
                int(pixmap.width() * self.zoom_factor),
                int(pixmap.height() * self.zoom_factor),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            label.setPixmap(scaled_pixmap)
            label.adjustSize()
        else:
            label.setText("无法加载图片")

    def update_zoom(self, value):
        """更新缩放因子并刷新图片显示"""
        self.zoom_factor = value / 100.0
        if self.current_image is not None:
            self.display_image(self.current_image, self.original_label)
        if self.result_image is not None:
            self.display_image(self.result_image, self.result_label)

    def closeEvent(self, event):
        """窗口关闭时清理资源"""
        self.ws_client.stop()
        self.ws_client.wait()
        super().closeEvent(event)

    def toggle_autoplay(self):
        if self.is_autoplaying:
            self.autoplay_timer.stop()
            self.btn_autoplay.setText("自动播放")
            self.is_autoplaying = False
        else:
            if len(self.image_files) > 0:
                self.send_for_detection()
                self.btn_autoplay.setText("停止播放")
                self.is_autoplaying = True

    def autoplay_next_image(self):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_current_image()
            self.send_for_detection()
        else:
            self.autoplay_timer.stop()
            self.btn_autoplay.setText("自动播放")
            self.is_autoplaying = False

    def update_display_time(self, text: str):
        '''自动播放时，修改每张图片的显示时间'''
        try:
            self.display_time = int(text)
        except ValueError:
            pass
    def send_user_input(self):
        user_text = self.user_input_line.text().strip()
        if not user_text:
            QMessageBox.information(self, "提示", "请输入有效的指令。")
            return

        self.update_status(f"\n[用户输入] {user_text}")
        self.user_input_line.setDisabled(True)  # 防止重复点击
        # Agent线程
        self.agent_thread = AgentWorker(self.agent, user_text, self.history)
        self.agent_thread.result_ready.connect(self.handle_agent_response)
        self.agent_thread.history_ready.connect(self.update_history)
        self.agent_thread.finished.connect(lambda: self.user_input_line.setDisabled(False))
        self.agent_thread.start()

    def handle_agent_response(self, output):
        self.update_status(f"\n[Agent响应] {output}")
        # Speech线程（会闪退：和Qt）
        # self.speak_text(output)
        self.user_input_line.clear()

    def update_history(self, new_history):
        self.history = new_history
    
    def speak_text(self, text: str):
        if hasattr(self, 'speech_thread') and self.speech_thread is not None:
            return  # 上一个播报还在进行中，跳过

        self.speech_thread = SpeechWorker(text)
        self.speech_thread.finished.connect(self.clear_speech_thread)
        self.speech_thread.start()

    def clear_speech_thread(self):
        self.speech_thread = None  # 清理线程引用

if __name__ == "__main__":
    # 示例配置，根据需要修改
    my_ip = "10.194.151.116"  # 上位机
    my_port = 8765
    server_uri = "ws://10.171.103.229:8765"  # jetson开发板

    app = QApplication(sys.argv)
    window = DetectionApp(my_ip, my_port, server_uri)
    window.show()
    sys.exit(app.exec_())
