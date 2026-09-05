# Person ReID Project — MVP-1

MVP-1 是本项目的检测基础版本：使用 Ultralytics YOLOv8 和
`weights/yolov8n.pt` 只检测 COCO `person` 类，并通过 OpenCV 显示检测框、类别和
置信度。

本阶段支持：

- 摄像头索引，例如 `0`；
- 本地视频文件；
- CUDA 可用时自动使用 CUDA，否则回退 CPU；
- Windows 默认 `num_workers=0`；
- 按 `q` 或 `Q` 退出。

本阶段暂不包含 BoT-SORT、Track ID、OSNet/ReID、TargetGallery、SQLite、ROI 选人
和 RTSP。它们会在后续 MVP 阶段实现。

## 环境

建议使用已经验证过的 Python 3.10 环境。安装 MVP-1 依赖：

```bash
python -m pip install -r requirements.txt
```

项目已包含 YOLO 权重：

```text
weights/yolov8n.pt
```

## 运行

默认读取摄像头 `0`：

```bash
python app.py
```

读取本地视频：

```bash
python app.py --source data/demo.mp4
```

也可以修改 `config/config.yaml` 中的 `video.source`。数字字符串会被解析为摄像头
索引，其他字符串按本地视频路径处理。

## 测试

测试不执行 GPU 推理，不要求打开摄像头：

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## 代码结构

```text
app.py                    # MVP-1 入口和主循环
config/config.yaml       # MVP-1 配置
src/config.py            # 配置和设备选择
src/video_source.py      # 摄像头/视频读取
src/detector.py          # YOLOv8n person-only 检测
src/models.py            # Detection 数据类
src/visualization.py      # 检测框和标签绘制
src/logging_utils.py     # logging 配置
ui/opencv_ui.py          # OpenCV 窗口和按键
tests/                   # 基础单元测试
```
