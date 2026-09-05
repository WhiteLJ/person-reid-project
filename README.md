# Person ReID Project — MVP-2

MVP-2 使用 Ultralytics YOLOv8 和 `weights/yolov8n.pt`，通过 Ultralytics BoT-SORT
为 COCO `person` 检测结果分配临时 Track ID。

本阶段支持：

- 摄像头索引，例如 `0`；
- 本地视频文件；
- YOLOv8 person detection + BoT-SORT temporary Track ID；
- 连续帧之间复用 tracker 状态；
- CUDA 可用时自动使用 CUDA，否则回退 CPU；
- Windows 默认 `num_workers=0`；
- 按 `q` 或 `Q` 退出。

本阶段暂不包含：

- 手动选人和 ROI；
- Torchreid/OSNet；
- ReID；
- Person ID；
- TargetGallery；
- SQLite；
- BoxMOT；
- RTSP。

MVP-2 的逐帧主链路只调用一次 `model.track(frame, ...)`，不对同一帧再次调用
`model.predict()`。`botsort.yaml` 使用 Ultralytics 内置配置，appearance ReID 保持关闭。

## 环境和依赖

建议使用已经验证过的 Python 3.10 环境。正式 MVP 依赖说明是
`requirements.txt`：

```bash
python -m pip install -r requirements.txt
```

`pip-freeze.txt` 仅是本机环境快照，不作为安装入口。

项目已包含正式模型权重：

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

## 人工验证 Track ID

1. 启动 `python app.py`，让一个人连续走动，观察框上的 `ID n` 是否保持不变。
2. 让第二个人进入画面并同时移动，确认两个人显示不同的 Track ID。
3. 短暂用物体遮挡其中一人，确认遮挡前后的 ID 是否尽量保持一致。
4. 不要以“离开画面后重新出现仍保持 ID”作为 MVP-2 验收标准，该能力由后续 ReID 实现。

## 代码结构

```text
app.py                      # MVP-2 入口和主循环
config/config.yaml          # MVP-2 配置
src/config.py               # 配置和设备选择
src/video_source.py         # 摄像头/视频读取
src/detector.py             # 保留的 MVP-1 检测模块
src/tracking_pipeline.py    # YOLOv8n + BoT-SORT
src/models.py               # Detection / Track 数据类
src/visualization.py        # 检测框和 Track ID 绘制
src/logging_utils.py        # logging 配置
ui/opencv_ui.py             # OpenCV 窗口和按键
tests/                      # MVP-1/MVP-2 单元测试
```
