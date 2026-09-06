# Person ReID Project — MVP-6

MVP-6 在 MVP-5 的基础上增加内存 `TargetGallery`。用户可以将已经通过 `S` 选择
的 SessionTarget 显式加入 Gallery，获得当前进程内的 `GalleryPerson`（如 P001）。
YOLOv8 使用 `weights/yolo/yolov8n.pt`，BoT-SORT 继续生成临时 Track ID；
`SessionTarget.target_id` 和 `GalleryPerson.person_id` 都不会持久化。

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
- 按 `q` 或 `Q` 退出；
- Torchreid / OSNet ReID feature extraction；
- normalized 512-D ReID embedding；
- cosine similarity validation；
- ACTIVE / LOST / RECOVER session target；
- 受 grace period、批量 ReID、threshold 和双侧 margin 约束的 Track ID 恢复；
- 受一致性阈值保护且有最大长度的 reference embedding bank；
- `G / g`：将已有 SessionTarget 连续加入内存 TargetGallery；
- 已入 Gallery 的目标显示 `TARGET P001 | ID n`。

按 `S`、`R` 或 `G` 后进入暂停编辑会话。编辑会话使用进入模式时冻结的当前帧和
`tracks` 列表，允许连续拖动多个 ROI；Enter/Space 结束会话，Esc 取消当前未完成
的拖框，Q 退出程序。编辑期间不会重新运行 YOLO 或 BoT-SORT，也不会读取下一帧。
每帧主链路仍然只调用一次 `model.track(frame, ...)`。

本阶段暂不包含：

- SQLite；
- Person ID（当前只有临时的 SessionTarget）；
- 持久化 Person ID；
- 跨程序启动后的身份恢复；
- 持久化 TargetGallery；
- 自动识别历史目标库；
- 训练或微调模型；
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

`pip-freeze.txt` 仅是本机环境快照，不作为安装入口。正式依赖中的 Torchreid
使用官方 Git 来源 `KaiyangZhou/deep-person-reid`，不使用 PyPI 上同名的旧包。
`requirements.txt` 中固定为当前已验证的官方 commit：

```text
torchreid @ git+https://github.com/KaiyangZhou/deep-person-reid.git@f8cd150fdf77e8d9e1ed143b7f308c2c609ded50
```

ReID checkpoint 必须由用户预先放置到：

```text
weights/reid/osnet_x0_25_msmt17.pth
```

程序不会在正常运行时联网下载 checkpoint，也不会回退到 ImageNet-only 权重。

项目已包含正式模型权重：

```text
weights/yolo/yolov8n.pt
```

MVP-5 的恢复参数位于 `reid_recovery` 配置节。当前值是保守的工程初值，需结合
实际视频调整：`lost_grace_frames=10`、`reference_update_interval_frames=15`、
`recovery_interval_frames=10`、`max_reference_embeddings=8`、
`recovery_threshold=0.75`、`recovery_margin=0.05`、
`reference_update_threshold=0.80`。恢复使用 normalized centroid similarity，
不会因单个异常历史 embedding 的高分直接绑定。

## ReID smoke test

准备三张图片：A1、A2 为同一个人，B1 为另一个人，然后运行：

```bash
python -m tools.reid_smoke_test A1.jpg A2.jpg B1.jpg
```

工具会输出每张图片的 embedding shape、dtype、L2 norm，以及
`cos(A1, A2)` 和 `cos(A1, B1)`。同一人与不同人的相似度关系仅作为当前素材的
验证结果。MVP-5 的恢复阈值、margin 和 reference 更新阈值只是可调工程初值，
不代表通用最优值。

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

## 人工验证 MVP-6

1. 启动程序，按 `S`；主窗口暂停，在主窗口中拖动第一个 ROI，完成后该目标应立即显示 `TARGET | ID n`。
2. 不要重新按 `S`，继续拖动第二个、第三个 ROI；多个目标应同时特殊高亮，重复框选同一人不会产生重复目标。
3. 按 `Enter` 或 `Space` 结束 ADD 会话并恢复视频；按 `Esc` 只取消当前未完成的拖框。
4. 按 `R` 进入 REMOVE 会话；框选一个或多个已选目标，只有对应目标取消高亮，其他目标不受影响。
5. 按 `R` 框选普通 Track 或空白区域；当前目标集合不应改变，并会记录日志提示。
6. 按 `C` 清除全部目标，所有 `TARGET` 特殊框消失；普通 Track 默认仍不显示，
   只有 `ui.show_unselected_tracks=true` 时才显示普通绿色 Track。
7. 目标短暂遮挡后，如果 BoT-SORT 恢复相同 Track ID，应继续特殊高亮，且不应发生
   不必要的重绑定。
8. 目标完全离开后，观察日志中的 `TARGET_LOST target=... old_track_id=...`；以新
   Track ID 返回并满足阈值/双侧 margin 后，应记录 `TARGET_RECOVERED`，红框继续跟随。
9. 在 A 离开期间让 B/C 活动，确认 A 不会恢复到已被 ACTIVE 目标占用的 Track。
10. 使用明显不同的人作为候选；相似度不足时目标保持 LOST，不发生误绑定。
11. 选择 A/B，单独让 A 离开再回来，确认只恢复 A，B 的 SessionTarget 不变化。
12. 目标离开后按 `R` 删除仍可见的其他目标或按 `C` 清除；删除/清除后不再执行其
    后续恢复。
13. 按 `G`，连续框选两个已经通过 `S` 选择的目标；确认分别生成 P001/P002，且
    不需要重新运行 ReID。
14. 再次框选同一个 SessionTarget，确认仍为原 GalleryPerson；框选普通 Track 时
    确认 Gallery 不变化并记录 `GALLERY_ENROLL_REJECTED`。
15. 删除 GalleryPerson 后，SessionTarget 仍能继续显示和跟踪；按 R/C 删除目标
    时，GalleryPerson 保留但 session-target 关联消失。

## 代码结构

```text
app.py                      # MVP-6 入口和主循环
config/config.yaml          # MVP-6 配置
src/config.py               # 配置和设备选择
src/video_source.py         # 摄像头/视频读取
src/detector.py             # 保留的 MVP-1 检测模块
src/tracking_pipeline.py    # YOLOv8n + BoT-SORT
src/reid.py                 # Torchreid OSNet embedding 提取
src/roi_selector.py         # ROI 与 Track 的 IoU 匹配
src/target_manager.py       # SessionTarget 状态和 Track 绑定
src/target_recovery.py      # reference bank 和 LOST/RECOVER 协调
src/gallery.py              # 内存 TargetGallery 和 enrollment 关联
src/models.py               # Detection / Track / SessionTarget 数据类
src/visualization.py        # 普通框和目标高亮绘制
src/logging_utils.py        # logging 配置
ui/opencv_ui.py             # OpenCV 窗口和按键 Action
ui/roi_editor.py            # 暂停编辑会话和鼠标拖框
tools/reid_smoke_test.py    # 三图 ReID embedding 验证工具
tests/                      # MVP-1 至 MVP-6 单元测试
```
