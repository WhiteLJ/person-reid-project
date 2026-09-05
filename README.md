# Person ReID Project — MVP-3

MVP-3 使用 Ultralytics YOLOv8 和 `weights/yolov8n.pt`，通过 Ultralytics BoT-SORT
生成临时 Track ID，并支持用户通过 OpenCV ROI 同时选择多个当前 Track。

本阶段支持：

- 摄像头索引，例如 `0`；
- 本地视频文件；
- YOLOv8 person detection + BoT-SORT temporary Track ID；
- 连续帧之间复用 tracker 状态；
- `S / s`：进入连续 ROI 会话并加入当前目标集合；
- `R / r`：进入连续 ROI 会话并从当前目标集合移除；
- `C / c`：清除所有当前目标；
- 选中目标使用特殊框和 `TARGET | ID n` 高亮；
- 默认不显示未选中的 Track；可通过 `ui.show_unselected_tracks: true` 开启绿色调试框；
- CUDA 可用时自动使用 CUDA，否则回退 CPU；
- Windows 默认 `num_workers=0`；
- 按 `q` 或 `Q` 退出。

按 `S` 或 `R` 后进入暂停编辑会话。编辑会话使用进入模式时冻结的当前帧和
`tracks` 列表，允许连续拖动多个 ROI；Enter/Space 结束会话，Esc 取消当前未完成
的拖框，Q 退出程序。编辑期间不会重新运行 YOLO 或 BoT-SORT，也不会读取下一帧。
每帧主链路仍然只调用一次 `model.track(frame, ...)`。

本阶段暂不包含：

- Torchreid/OSNet；
- ReID embedding；
- 离开画面后的身份恢复；
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

`pip-freeze.txt` 仅是本机环境快照，不作为安装入口。MVP-3 不安装、不加载、也不
调用 Torchreid/OSNet。

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

## 人工验证多目标选择

1. 启动程序，按 `S`；主窗口暂停，在主窗口中拖动第一个 ROI，完成后该目标应立即显示 `TARGET | ID n`。
2. 不要重新按 `S`，继续拖动第二个、第三个 ROI；多个目标应同时特殊高亮，重复框选同一人不会产生重复目标。
3. 按 `Enter` 或 `Space` 结束 ADD 会话并恢复视频；按 `Esc` 只取消当前未完成的拖框。
4. 按 `R` 进入 REMOVE 会话；框选一个或多个已选目标，只有对应目标取消高亮，其他目标不受影响。
5. 按 `R` 框选普通 Track 或空白区域；当前目标集合不应改变，并会记录日志提示。
6. 按 `C` 清除全部目标，所有 Track 恢复普通显示。
7. 目标短暂遮挡后，如果 BoT-SORT 恢复相同 Track ID，应继续特殊高亮。
8. 目标完全离开后以新 Track ID 返回时，MVP-3.1 不自动重新绑定；该能力留给后续 ReID。

## 代码结构

```text
app.py                      # MVP-3 入口和主循环
config/config.yaml          # MVP-3 配置
src/config.py               # 配置和设备选择
src/video_source.py         # 摄像头/视频读取
src/detector.py             # 保留的 MVP-1 检测模块
src/tracking_pipeline.py    # YOLOv8n + BoT-SORT
src/roi_selector.py         # ROI 与 Track 的 IoU 匹配
src/target_manager.py       # 多目标 Track ID 选择状态
src/models.py               # Detection / Track 数据类
src/visualization.py        # 普通框和目标高亮绘制
src/logging_utils.py        # logging 配置
ui/opencv_ui.py             # OpenCV 窗口和按键 Action
ui/roi_editor.py            # 暂停编辑会话和鼠标拖框
tests/                      # MVP-1/MVP-2/MVP-3/MVP-3.1 单元测试
```
