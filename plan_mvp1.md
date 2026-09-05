# MVP-1 实施计划

## 目标

完成 YOLOv8n 行人检测最小可运行版本：支持摄像头和本地视频文件输入，使用
`weights/yolov8n.pt` 只检测 COCO `person` 类，使用 OpenCV 显示检测框、类别和
置信度，自动选择 CUDA/CPU，Windows 默认 `num_workers=0`，按 `q` 退出，并具备
基本日志、异常处理和基础测试。

## 本阶段边界

本阶段不实现：

- Ultralytics BoT-SORT 和 Track ID；
- Torchreid 1.4.0、OSNet `osnet_x0_25` 和 ReID embedding；
- TargetManager、TargetGallery 和 Person ID；
- SQLite 持久化；
- ROI 选人、入库快捷键和复杂 UI；
- RTSP 输入（安排在 MVP-9）。

## 已确认的文件变更

已修改：

- `AGENTS.md`：固定最终技术路线，明确 MVP-1 边界和配置要求。
- `person_reid_project_spec.md`：固定 YOLOv8n/Torchreid 版本与模型，补充 MVP-1
  验收标准，修正摄像头、RTSP 和权重路径说明。

## MVP-1 文件清单

创建：

- `README.md`：安装、运行、输入源示例、操作说明和阶段限制。
- `requirements.txt`：精简固定运行依赖。
- `config/config.yaml`：输入、YOLO、运行时和 OpenCV 显示配置。
- `app.py`：主循环、异常处理和资源清理。
- `src/__init__.py`：Python 包标记。
- `src/config.py`：YAML 配置读取、路径解析和 CUDA/CPU 自动选择。
- `src/models.py`：`Detection` 数据类。
- `src/video_source.py`：摄像头和本地视频输入封装。
- `src/detector.py`：YOLOv8n 单次加载、person-only 检测和结果解析。
- `src/visualization.py`：检测框、类别名和置信度绘制。
- `src/logging_utils.py`：统一日志配置。
- `ui/__init__.py`：Python 包标记。
- `ui/opencv_ui.py`：OpenCV 窗口显示和 `q` 退出处理。
- `tests/test_config.py`：配置加载和设备解析测试。
- `tests/test_video_source.py`：输入源解析测试。
- `tests/test_detection.py`：检测结果解析和 person 过滤测试。
- `tests/test_visualization.py`：检测绘图基本测试。

不移动、不复制已有的 `weights/yolov8n.pt`，不修改 `references/`。

## 实现步骤

1. 创建基础 Python 包、配置、README 和精简 requirements。
2. 实现 YAML 配置加载、相对项目根目录路径解析和自动设备选择。
3. 实现 OpenCV 摄像头/本地视频读取及资源释放。
4. 实现 YOLOv8n 检测器：加载一次，调用检测接口，传入 `classes=[0]`，不调用
   `model.track()`。
5. 实现检测结果数据类和解析逻辑，只返回 `person` 检测。
6. 实现 OpenCV 框、`person`、置信度显示及 `q` 退出。
7. 在主循环加入启动、模型、输入、结束和异常日志，并保证 `finally` 释放资源。
8. 使用不依赖 GPU/模型权重的伪造结果编写基础测试。
9. 运行单元测试、帮助命令和一次最小模型/视频推理检查。

## 验收标准

- `python app.py` 可读取默认摄像头或配置中的视频文件。
- 可通过配置或命令行切换摄像头索引与本地视频路径。
- 画面只显示 `person` 框、类别名和置信度，不显示 Track ID 或 Person ID。
- CUDA 可用时使用 CUDA，否则回退 CPU。
- Windows 默认 `num_workers=0`。
- 视频结束或按 `q` 后正常退出并释放摄像头、视频和窗口资源。
- 模型、输入源或帧读取错误会记录日志并以清晰异常结束。
- 基础测试可在无 GPU、无模型推理的情况下运行。
