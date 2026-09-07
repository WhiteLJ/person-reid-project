# Person ReID Project — MVP-7

本项目是一个基于 YOLOv8、Ultralytics BoT-SORT、Torchreid/OSNet 和 OpenCV 的
交互式行人跟踪与重识别工程原型。

MVP-7 在 MVP-6 的内存 `TargetGallery` 基础上增加 SQLite 持久化。当前严格区分：

- `Track ID`：BoT-SORT 生成的临时轨迹身份；
- `SessionTarget.target_id`：当前进程内的临时目标身份；
- `GalleryPerson.person_id`：目标库中的长期逻辑身份，例如 `P001`。

## 当前支持

- 摄像头和本地视频输入；
- YOLOv8 person detection + Ultralytics BoT-SORT temporary Track ID；
- OpenCV 多目标 ROI 选择、删除和清除；
- Torchreid 1.4.0 / OSNet `osnet_x0_25` ReID embedding；
- ACTIVE / LOST / RECOVER SessionTarget；
- 通过 ReID 恢复发生 Track ID 变化的当前目标；
- `G` 显式将已有 SessionTarget 加入 Gallery 并持久化；
- `GalleryPerson`、reference embeddings 和 centroid 的 SQLite 持久化；
- 程序重启后保留 Gallery person ID、label 和特征；
- 离线 `gallery_admin` 管理工具。

SQLite 默认路径来自配置：

```yaml
database:
  path: "database/person_reid.db"
```

Repository 会自动创建数据库父目录。数据库中不保存 Track、SessionTarget、ACTIVE/LOST
状态或 session-target 映射；程序重启后只恢复 GalleryPerson。已经加载的 Gallery
不会在 MVP-7 中自动识别普通 Track，也不会触发额外 ReID。

## 仍不支持

- Gallery 自动识别普通 Track（MVP-8）；
- 程序重启后自动绑定新的 SessionTarget；
- SQLite 中保存 SessionTarget 或 Track 状态；
- 最终 PySide/Qt Gallery 管理界面；
- RTSP、训练或微调模型、BoxMOT。

最终 UI 需求（本阶段不实现）包括：当前红框目标直接“添加到库”、已入库目标“从库中
移除”、独立目标库侧栏、查看人员、单个/批量删除、清空和修改 label。

## 环境和依赖

正式安装入口是 `requirements.txt`：

```bash
python -m pip install -r requirements.txt
```

`pip-freeze.txt` 仅为本机环境快照，不是安装入口。Torchreid 使用官方 Git 来源
`KaiyangZhou/deep-person-reid`，不使用 PyPI 上同名旧包；项目不会因此升级当前
PyTorch、CUDA 或 Ultralytics。

运行前请准备：

```text
weights/yolo/yolov8n.pt
weights/reid/osnet_x0_25_msmt17.pth
```

程序不会在正常运行时联网下载 ReID checkpoint，也不会回退到 ImageNet-only 权重。

## 运行

默认读取摄像头 `0`：

```bash
python app.py
```

读取本地视频：

```bash
python app.py --source data/demo.mp4
```

常用按键：

- `S`：选择当前 SessionTarget；
- `R`：删除当前 SessionTarget；
- `G`：将已选择的 SessionTarget 显式加入 Gallery 并持久化，不重新运行 ReID；
- `C`：清除当前 SessionTargets，但不删除 GalleryPerson；
- `Q`：退出。

## Gallery 管理工具

`gallery_admin.py` 是 MVP-7 的离线开发/验收工具：

```bash
python -m tools.gallery_admin list
python -m tools.gallery_admin remove P001
python -m tools.gallery_admin clear
```

也可以指定临时数据库：

```bash
python -m tools.gallery_admin --db path/to/test.db list
```

执行 `remove`、`clear` 等修改操作前必须关闭正在运行的主程序，避免 SQLite 数据与
主程序内存中的 `TargetGallery` 不同步。删除 GalleryPerson 不会删除当前进程中的
SessionTarget；主程序重启后会按数据库内容重新加载 Gallery。

## ReID smoke test

准备 A1、A2（同一个人）和 B1（另一个人）：

```bash
python -m tools.reid_smoke_test A1.jpg A2.jpg B1.jpg
```

工具输出 embedding shape、dtype、L2 norm 以及 cosine similarity。MVP-5 的恢复阈值
和 margin 都是可调工程初值，不代表通用最优值。

## 测试

测试不启动真实摄像头或 GPU 推理；SQLite 测试使用临时数据库：

```bash
python -m unittest discover -s tests -p "test_*.py"
```

### 代码结构

```text
app.py                      # MVP-7 主入口和视频循环
config/config.yaml          # 应用配置
src/config.py               # 配置和设备选择
src/tracking_pipeline.py    # YOLOv8n + BoT-SORT
src/reid.py                 # OSNet embedding 提取
src/target_recovery.py      # SessionTarget LOST/RECOVER
src/gallery.py              # 纯内存 TargetGallery 业务层
src/gallery_service.py      # 内存 Gallery 与持久化协调
src/database.py             # SQLite GalleryRepository
src/models.py               # Detection / Track / SessionTarget
src/visualization.py        # Track 和目标绘制
ui/roi_editor.py            # OpenCV 暂停 ROI 编辑会话
tools/gallery_admin.py      # 离线 Gallery 管理工具
tests/                      # MVP-1 至 MVP-7 测试
```
