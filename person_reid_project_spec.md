# 行人目标锁定、重识别与目标库系统——项目设计说明 V2

> **项目定位**：企业竞赛作品 / 工程原型，不以算法创新和论文研究为目标。  
> **主要目标**：尽量复用成熟开源项目，以较低开发成本完整实现“手动选人—持续跟踪—离开后重识别—目标入库—再次出现自动识别”。  
> **开发协作方式**：后续主要与 Codex 协作编码；本文件作为需求和架构的统一依据。  
> **推荐语言**：Python 3.10+。  
> **最终固定技术路线**：Ultralytics YOLOv8（第一版权重 `yolov8n.pt`）+ Ultralytics BoT-SORT + Torchreid 1.4.0 / OSNet（第一版模型 `osnet_x0_25`）+ 自研 TargetGallery + SQLite + OpenCV。  
> **MVP-1 范围**：只实现 YOLOv8n 行人检测、摄像头/本地视频输入、OpenCV 检测结果显示和基础运行保障；不实现 BoT-SORT、OSNet/ReID、TargetGallery、SQLite 或复杂 UI。  
> **备用增强路线**：当 Ultralytics 内置 BoT-SORT 在复杂场景效果不足时，再切换到 BoxMOT。

---

## 1. 项目需求

系统接入摄像头、RTSP 视频流或本地视频后，应实时检测并跟踪画面中的行人。

### 1.1 手动目标锁定

用户可以在当前画面中手动框选任意一个人，系统应：

1. 判断用户框选区域对应哪一个当前行人 Track；
2. 将其设为“当前关注目标”；
3. 持续绘制特殊目标框；
4. 在连续运动或短时遮挡情况下尽量保持目标锁定；
5. 如果目标完全离开画面，保留其身份特征；
6. 当目标重新进入画面时，通过 Person ReID 自动重新认出并恢复锁定，即使 Tracker 已经分配了新的 Track ID。

### 1.2 目标库

用户可以将当前锁定目标主动加入目标库。

入库后，系统应持久化保存该人的：

- Person ID；
- 名称/别名（可选）；
- 缩略图；
- 多个 ReID embedding；
- centroid/代表特征；
- 创建时间；
- 最近出现时间；
- 备注（可选）。

之后程序重新启动时，目标库仍然存在。

### 1.3 自动识别目标库人员

目标库中的人员再次出现在摄像头范围内时，系统应自动：

1. 对该 Track 提取 ReID 特征；
2. 与目标库比对；
3. 如果相似度超过阈值，绑定到已有 Person ID；
4. 绘制醒目标识；
5. 显示 Person ID、Track ID、相似度等信息。

### 1.4 普通路人

普通路人只进行检测和临时跟踪，**不应自动加入永久目标库**。

这是本系统与很多“自动为所有人创建 Global ID”Demo 的关键区别。

---

## 2. 非目标 / 第一版不做的内容

为了控制开发复杂度，第一版明确不做：

- 自研目标检测算法；
- 自研 MOT 算法；
- 自研 ReID 网络；
- 论文级算法创新；
- 从零训练 YOLO；
- 从零训练 OSNet；
- 大规模向量数据库；
- 多摄像头跨镜联合跟踪；
- 人脸识别融合；
- 复杂 Qt 管理后台；
- 云服务或分布式部署。

以上功能如果竞赛后期确有需要，再作为增强项加入。

---

## 3. 三层身份必须严格区分

### 3.1 Track ID

Track ID 是 Tracker 产生的短期身份。

例如：

```text
Person A 当前：Track ID = 7
```

Person A 长时间离开画面后再次进入，可能变成：

```text
Track ID = 26
```

因此 Track ID 只能用于连续视频中的短时关联。

### 3.2 Person ID

Person ID 是系统自己维护的长期身份，例如：

```text
第一次出现：
Track ID = 7
Person ID = P001

离开画面

再次出现：
Track ID = 26
Person ID = P001
```

Person ID 依靠 ReID + Target Gallery 维护。

> **核心原则：Track ID 负责短时连续性；Person ID 负责长期身份。**

### 3.3 SessionTarget 与 GalleryPerson

MVP-5 引入的 `SessionTarget.target_id` 是当前程序运行期的临时目标身份，用于
维护 ACTIVE/LOST/RECOVER 和当前 Track 绑定。MVP-6 引入的 `GalleryPerson.person_id`
是用户显式 enrollment 后产生的目标库身份，例如 P001；MVP-7 起其特征和长期 ID
由 SQLite 持久化。SessionTarget 仍然只属于当前进程，不跨程序恢复。

```text
Track 17 -> SessionTarget 3 -> GalleryPerson P001
Track 17 -> LOST -> Track 29
                         └──仍然是 SessionTarget 3 / GalleryPerson P001
```

Gallery 自己维护 `session_target_id -> person_id` 映射。Track ID 改变不能改变
Gallery 身份；删除 SessionTarget 只解除映射，不删除 GalleryPerson。

---

## 4. 总体架构

```text
                  Camera / RTSP / Video
                           │
                           ▼
                  ┌─────────────────┐
                  │ Ultralytics YOLO│
                  │ Person Detection│
                  └────────┬────────┘
                           │
                      detections
                           │
                           ▼
                  ┌─────────────────┐
                  │    BoT-SORT     │
                  │ Short-term MOT  │
                  └────────┬────────┘
                           │
                    bbox + track_id
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
      UI/手动框选      ReID Extractor      Track Cache
          │             OSNet             临时状态
          │                │
          └──────────┬─────┘
                     ▼
               Target Manager
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
  当前临时锁定目标           Target Gallery
                                  │
                          embeddings + centroid
                                  │
                                  ▼
                               SQLite
                                  │
                                  ▼
                          Persistent Person ID
```

---

## 5. 技术选型

### 5.1 YOLO：Ultralytics

GitHub：

- https://github.com/ultralytics/ultralytics

用途：

- 行人检测；
- 本地视频、摄像头、RTSP 输入；
- 内置 Tracking API；
- 从 MVP-2 起接入其 BoT-SORT；MVP-1 只调用检测接口。

建议：

- 第一版固定使用 YOLOv8 的 `yolov8n.pt` COCO 预训练权重；
- 只保留 `person` 类；
- 后续如确有性能需求再评估其他 YOLOv8 规格，MVP-1 不切换模型规格；
- 暂不训练自定义检测模型。

---

### 5.2 Tracker：优先 Ultralytics BoT-SORT

第一版选择：

```text
Ultralytics YOLO + Ultralytics 内置 BoT-SORT
```

原因：

- 集成简单；
- 减少第三方框架层数；
- 满足普通竞赛演示中的连续跟踪需求；
- 后续仍可替换。

BoT-SORT 主要解决：

- 连续帧目标关联；
- 短时遮挡；
- 运动预测；
- Track ID 的连续性。

但必须注意：

> BoT-SORT 不是永久身份库，目标长时间离开画面后仍需依赖 ReID 重新识别。

---

### 5.3 Tracker 备用方案：BoxMOT

GitHub：

- https://github.com/mikel-brostrom/boxmot

BoxMOT 当前**不作为第一版必需依赖**，而作为 tracker 增强和源码参考。

如果实际测试出现以下情况，可切换：

- 多人交叉时 ID Switch 明显；
- 遮挡恢复较差；
- 希望测试 DeepOCSORT、StrongSORT 等其他方案；
- 希望更灵活地替换 tracker 和 ReID。

因此代码层应对 Tracker 做一层封装，避免业务逻辑绑定某一个实现。

---

### 5.4 Person ReID：OSNet / Torchreid

GitHub：

- https://github.com/KaiyangZhou/deep-person-reid

推荐模型：

```text
osnet_x0_25
```

固定使用官方 Torchreid 1.4.0；该组件从 MVP-4 开始接入，MVP-1 至 MVP-3.1
不加载 ReID 模型。

如果 GPU 性能充足，可换更大 OSNet。

职责：

```text
person crop -> normalized embedding
```

OSNet 专门针对 Person Re-Identification，比直接拿 ImageNet ResNet50 当人员特征提取器更合适。

MVP-4 固定使用官方 Model Zoo 的 MSMT17 `combineall` ReID checkpoint：

```text
原始文件名：
osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth

项目本地路径：
weights/reid/osnet_x0_25_msmt17.pth
```

该 checkpoint 是 Person ReID 训练权重，不等同于 ImageNet-only pretrained
weights。运行时必须使用本地 checkpoint；文件缺失或 backbone 参数无法匹配时
应明确报错，不得静默联网下载或回退到 ImageNet 权重。

---

## 6. 为什么不是“只用 YOLO”

```text
YOLO       = 这个位置是不是一个人？
BoT-SORT   = 这一帧的人是不是上一帧那个 Track？
OSNet      = 两张人物图像是不是可能属于同一个人？
Gallery    = 这个人是不是系统以前保存的 P001/P002？
```

因此需求完整实现至少需要：

```text
Detection + Tracking + ReID + Identity Memory
```

---

## 7. GitHub 参考仓库体系

建议将所有参考项目下载到主项目的：

```text
references/
```

目录下。

### 7.1 仓库分级

| 仓库 | 定位 | 是否运行时依赖 | Codex 重点学习内容 |
|---|---|---:|---|
| `ultralytics` | 核心框架 | 是（通常 pip 安装） | YOLO、BoT-SORT、Results/Track API |
| `deep-person-reid` | 核心 ReID（Torchreid 1.4.0） | 是/间接使用 | OSNet、FeatureExtractor、预处理 |
| `Re-id_initial` | 小型完整参考 Demo | 否 | PersonGallery、OSNet、tracking+ReID 串联 |
| `person-reid-yolov8-tracking` | Global ID 业务参考 | 否 | 多 embedding、centroid、re-entry、identity memory |
| `person_reid_yolo` | 持久化参考 | 否 | SQLite 人员库、跨视频身份加载 |
| `boxmot` | Tracker 增强/备用 | 默认否 | 统一 tracker 接口、复杂 MOT + ReID |

---

### 7.2 Ultralytics

https://github.com/ultralytics/ultralytics

重点：

- YOLO 推理；
- `model.track()`；
- `persist=True`；
- `Results.boxes.id`；
- `botsort.yaml`；
- 视频/摄像头输入；
- track 结果解析。

主项目运行时通常通过：

```bash
pip install ultralytics
```

使用；`references/ultralytics` 主要供 Codex 查源码。

---

### 7.3 deep-person-reid / Torchreid

https://github.com/KaiyangZhou/deep-person-reid

重点：

- OSNet 模型结构；
- FeatureExtractor；
- 图像 resize/normalize；
- embedding 输出；
- 预训练权重加载。

---

### 7.4 Re-id_initial

https://github.com/KunalNath04/Re-id_initial

这是一个很适合 Codex 快速理解完整业务链的小项目。

重点文件：

```text
gallery.py
reid_model.py
object_tracking_reid.py
```

重点学习：

```text
YOLO
  ↓
Tracker
  ↓
Person crop
  ↓
OSNet
  ↓
embedding
  ↓
PersonGallery
  ↓
cosine matching
```

本项目不会直接依赖该仓库，但其 Gallery 和 ReID 调用方式非常值得参考。

---

### 7.5 person-reid-yolov8-tracking

https://github.com/Megha54049/person-reid-yolov8-tracking

最值得参考：

- `GlobalIdentityManager`；
- 一个身份保存多个 embedding；
- centroid；
- 最近 embedding 比较；
- re-entry；
- identity memory。

不建议照搬：

- Simple IoU Tracker；
- ImageNet ResNet50 作为最终 ReID；
- 固定视频/477 帧逻辑；
- 自动给所有路人创建永久 Global ID。

本项目应把其：

```text
GlobalIdentityManager
```

思想改成：

```text
TargetGallery
```

并且**只有用户主动入库的人才获得永久 Person ID**。

---

### 7.6 person_reid_yolo

https://github.com/VladimirSinitsin/person_reid_yolo

主要参考：

- 人员信息持久化；
- SQLite 数据库结构；
- 程序重启后重新加载人员数据；
- 后续视频再次识别。

不重点参考其较旧 YOLO 实现。

---

### 7.7 BoxMOT

https://github.com/mikel-brostrom/boxmot

主要参考：

- tracker 抽象方式；
- detector/tracker/ReID 的解耦；
- BoT-SORT、DeepOCSORT、StrongSORT 等；
- ReID 和 MOT 的工程集成方式。

只有 Ultralytics 内置 BoT-SORT 实测效果不够时，才考虑引入主项目。

---

## 8. 参考仓库下载方式

推荐项目结构：

```text
person-reid-project/
│
├── AGENTS.md
├── README.md
├── person_reid_project_spec.md
├── src/
├── config/
├── tests/
├── data/
├── weights/
│
└── references/
    ├── ultralytics/
    ├── deep-person-reid/
    ├── Re-id_initial/
    ├── person-reid-yolov8-tracking/
    ├── person_reid_yolo/
    └── boxmot/
```

下载命令：

```bash
mkdir references
cd references

git clone --depth 1 https://github.com/ultralytics/ultralytics.git
git clone --depth 1 https://github.com/KaiyangZhou/deep-person-reid.git
git clone --depth 1 https://github.com/KunalNath04/Re-id_initial.git
git clone --depth 1 https://github.com/Megha54049/person-reid-yolov8-tracking.git
git clone --depth 1 https://github.com/VladimirSinitsin/person_reid_yolo.git
git clone --depth 1 https://github.com/mikel-brostrom/boxmot.git
```

`--depth 1` 只保留当前源码，不下载完整 Git 历史，足够 Codex 阅读。

### references 目录规则

1. `references/` 只用于阅读和借鉴；
2. 不在其中直接开发本项目功能；
3. 不修改第三方仓库；
4. 主项目代码全部写到 `src/`、`ui/`、`tests/`；
5. 如果需要参考某个实现，让 Codex 明确指出参考了哪个文件/思想；
6. 不要把多个仓库的代码无规划拼接到一个文件。

---

## 9. 核心业务流程

### 9.1 普通检测与跟踪

MVP-1 在 YOLO person detection 后停止；从 MVP-2 才接入 BoT-SORT 并输出
temporary Track ID。以下是最终完整链路。

每帧：

```text
frame
 ↓
YOLO person detection
 ↓
BoT-SORT
 ↓
Track(track_id, bbox, conf)
```

普通人员只显示 Track ID，不生成永久 Person ID。

---

### 9.2 用户手动框选目标

MVP-3.1 使用 OpenCV 主窗口 mouse callback 的暂停编辑会话，不使用
`cv2.selectROI()` 或独立 ROI 窗口。

逻辑：

```text
用户 ROI
  ↓
与当前 tracks 的 bbox 计算 IoU
  ↓
选择最大 IoU Track
  ↓
TargetManager.select(...) 或 TargetManager.deselect(...)
```

如果最大 IoU 过低，则视为没有选中有效人物。
MVP-3 只保存临时 `Track ID`，不提取 crop、不运行 OSNet/ReID，也不创建 Person ID；
MVP-5 的选择入口在当前 Track crop 上提取首个 ReID reference，并创建仅限当前
进程的 `SessionTarget`。

---

### 9.3 当前目标 TRACKING

如果某个 SessionTarget 的 `current_track_id` 仍然存在：

- 直接跟随该 Track；
- 绘制特殊颜色框；
- 当前 Track 暂时消失时保留 `last_track_id`；
- 如果 BoT-SORT 恢复相同 Track ID，则继续高亮；
- MVP-5 只按配置间隔为 ACTIVE 目标低频采样，并在一致性阈值通过时更新
  bounded reference bank。

---

### 9.4 当前目标 LOST

当连续若干帧没有找到 SessionTarget 的 `current_track_id`，且超过配置的 grace
period：

```text
TRACKING -> LOST
```

保留：

```text
target_id（仅当前进程）
last_track_id
reference_embeddings
centroid
missing_frames
```

然后按 recovery interval 检查未被 ACTIVE target 占用的 candidate Tracks；没有
足够相似或 margin 不足时保持 LOST，不强制绑定。

---

### 9.5 目标重新进入

对于新 Track，且存在 LOST SessionTarget：

```text
new track
 ↓
crop
 ↓
OSNet
 ↓
query embedding
 ↓
与 LOST target 的 normalized centroid 比较
```

达到阈值：

```text
new_track_id -> 原 SessionTarget
LOST -> ACTIVE
```

如果当前目标已有后续阶段的 Person ID：

```text
P001 -> new_track_id
```

---

### 9.6 加入目标库

用户点击“加入目标库”后：

1. 生成 `P001/P002/...`；
2. 保存当前高质量缩略图；
3. 保存多个当前 reference embeddings；
4. 建立 centroid；
5. 写入 SQLite；
6. 当前 target 与该 Person ID 绑定。

不要只保存一张图片/一个 embedding。

---

### 9.7 自动识别目标库人员

推荐对以下 Track 执行 Gallery Search：

- 新 Track；
- 尚未绑定 Person ID 的 Track；
- 之前匹配不确定、到达重试间隔的 Track；
- LOST target 搜索阶段。

流程：

```text
Track crop
  ↓
OSNet embedding
  ↓
TargetGallery.search()
  ↓
best_person_id + score
  ↓
score >= threshold ?
  ↓ yes
bind(track_id, person_id)
```

---

## 10. Target Gallery 设计

建议自己实现：

```python
class TargetGallery:
    def add_person(...): ...
    def remove_person(...): ...
    def add_embedding(...): ...
    def search(...): ...
    def get_person(...): ...
    def list_persons(...): ...
    def load(...): ...
```

### 10.1 内存结构

```python
{
    "P001": {
        "name": "target_001",
        "embeddings": [...],
        "centroid": ...,
        "thumbnail_path": "...",
        "created_at": "...",
        "last_seen_at": "..."
    }
}
```

### 10.2 多 embedding

每个人建议保存 20~100 个上限，第一版可设：

```text
max_embeddings_per_person = 50
```

来源尽量覆盖：

- 正面；
- 背面；
- 左右侧；
- 不同时刻；
- 不同人物尺度。

### 10.3 Centroid

```text
centroid = normalize(mean(embeddings))
```

第一版搜索可以简单使用：

```text
score = max cosine(query, gallery_embeddings)
```

或者：

```text
score = max(
    cosine(query, centroid),
    max cosine(query, recent_embeddings)
)
```

不需要一开始设计复杂融合公式。

---

## 11. ReID 特征质量控制

错误 embedding 会污染 Gallery，因此只在人物 crop 质量较好时更新。

建议基本条件：

```text
bbox_width >= min_width
bbox_height >= min_height
detection_conf >= min_conf
bbox 未严重出界
距离上次保存 >= embedding_update_interval
```

可选后续条件：

- Laplacian 清晰度；
- 遮挡比例；
- 姿态质量；
- bbox 面积。

第一版不要过度复杂化，可先“每 20 帧最多保存一次”。

---

## 12. ReID 相似度

OSNet embedding 输出后统一 L2 Normalize：

```python
embedding = embedding / np.linalg.norm(embedding)
```

Cosine Similarity：

```text
score = query @ gallery_embedding
```

阈值必须配置化。

初始可从类似以下范围开始试：

```text
gallery_threshold: 0.70
selected_target_threshold: 0.70
```

但这些只是工程初始值，最终应根据竞赛现场/甲方摄像头视频调参。

---

## 13. SQLite 设计

### 13.1 persons

```sql
CREATE TABLE persons (
    person_id TEXT PRIMARY KEY,
    name TEXT,
    thumbnail_path TEXT,
    created_at TEXT NOT NULL,
    last_seen_at TEXT,
    note TEXT
);
```

### 13.2 embeddings

```sql
CREATE TABLE embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT NOT NULL,
    embedding BLOB NOT NULL,
    created_at TEXT NOT NULL,
    quality REAL,
    FOREIGN KEY(person_id) REFERENCES persons(person_id)
);
```

第一版可直接将 float32 numpy embedding 转 bytes 保存 BLOB。

也可以选择：

```text
SQLite 保存 metadata
.npy/.npz 保存 embedding
```

但从项目整洁性来看，第一版直接 SQLite BLOB 就够用。

---

## 14. TargetManager 状态机

MVP-5 的 `TargetManager` 管理多个当前会话目标；这里的 `target_id` 只在本次
程序运行期间有效，不是 Person ID。每个 SessionTarget 独立维护自己的 Track
绑定、状态和 ReID reference bank。MVP-5 的实际状态流转为：

```text
SessionTarget 1 -> current Track 3 -> ACTIVE
SessionTarget 2 -> current Track 8 -> ACTIVE

SessionTarget 1 -> current Track None, last Track 3 -> LOST
                         │
                         │ ReID recovery
                         ▼
                 current Track 11 -> ACTIVE

用户取消某个目标：删除整个 SessionTarget 及其 references
用户清除全部：删除全部 SessionTargets 及其 references
```

SessionTarget 至少包含：

```text
target_id
current_track_id
last_track_id
state = ACTIVE / LOST
reference_embeddings
normalized centroid
missing_frames
```

短暂缺失只增加 `missing_frames`；达到配置的 grace period 后才进入 LOST。恢复
必须经过 ReID threshold 和双侧 margin，不满足条件时保持 LOST。MVP-5 不在
SessionTarget 中加入 Person ID、Gallery ID 或持久化字段。

---

## 15. 代码目录结构

```text
person-reid-project/
│
├── AGENTS.md
├── README.md
├── person_reid_project_spec.md
├── requirements.txt
├── app.py
│
├── config/
│   ├── config.yaml
│   └── botsort.yaml           # 如果需要自定义
│
├── src/
│   ├── __init__.py
│   ├── models.py              # Detection / Track / GalleryMatch 等数据类
│   ├── tracking_pipeline.py   # YOLO + Tracker 封装
│   ├── reid.py                # OSNet FeatureExtractor
│   ├── gallery.py             # TargetGallery
│   ├── target_manager.py      # 当前锁定目标状态机
│   ├── database.py            # SQLite Repository
│   ├── video_source.py        # Video / Camera / RTSP
│   ├── visualization.py
│   ├── roi_selector.py
│   └── utils.py
│
├── ui/
│   ├── opencv_ui.py
│   └── qt_ui.py               # 后期可选
│
├── data/
│   ├── gallery.db
│   └── thumbnails/
│
├── weights/
│   ├── yolo/
│   └── reid/
│
├── tests/
│   ├── test_gallery.py
│   ├── test_database.py
│   ├── test_roi_match.py
│   ├── test_reid_similarity.py
│   └── test_target_manager.py
│
└── references/
    ├── ultralytics/
    ├── deep-person-reid/
    ├── Re-id_initial/
    ├── person-reid-yolov8-tracking/
    ├── person_reid_yolo/
    └── boxmot/
```

上述目录按 MVP 阶段逐步落地。MVP-1 复用现有 `weights/yolo/yolov8n.pt`，只创建检测
所需的基础模块；ReID、Gallery、TargetManager、SQLite 和复杂 UI 模块在对应阶段再加入。

---

## 16. 推荐数据类

避免全项目用裸 list/dict 传递 bbox。

```python
@dataclass
class Detection:
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int

@dataclass
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    person_id: str | None = None
    similarity: float | None = None
```

上面的 `person_id` 和 `similarity` 字段属于后续 Person ID / Gallery 阶段的设计；
MVP-4 的实际 `Track` 数据结构仍只包含 `track_id`、`bbox`、`confidence` 和
`class_id`。ReID embedding 独立于 `Track` 保存和传递。

可再增加：

```python
@dataclass
class GalleryMatch:
    person_id: str | None
    score: float
```

---

## 17. 核心模块接口

### 17.1 TrackingPipeline

```python
class TrackingPipeline:
    def process(self, frame) -> list[Track]:
        ...
```

第一版内部使用 Ultralytics；后面若切 BoxMOT，外层接口不变。

### 17.2 ReIDExtractor

```python
class ReIDExtractor:
    def extract(self, crop) -> np.ndarray:
        ...

    def extract_batch(self, crops) -> np.ndarray:
        ...
```

### 17.3 TargetGallery

```python
class TargetGallery:
    def add_person(self, embeddings, thumbnail=None, name=None) -> str:
        ...

    def add_embedding(self, person_id, embedding, quality=None):
        ...

    def search(self, embedding) -> GalleryMatch:
        ...

    def delete_person(self, person_id):
        ...
```

### 17.4 TargetManager

```python
class TargetManager:
    def select(self, track, embedding, frame_index=0) -> SessionTarget: ...
    def deselect(self, track) -> bool: ...
    def update_visibility(self, tracks, lost_grace_frames, frame_index): ...
    def add_reference(self, target_id, embedding, ...): ...
    def recover(self, target_id, track, embedding, ...): ...
    def clear(self): ...
```

### 17.5 DatabaseRepository

```python
class DatabaseRepository:
    def create_person(...): ...
    def save_embedding(...): ...
    def load_all_persons(...): ...
    def delete_person(...): ...
```

---

## 18. Track 与 Person 的运行时绑定

建议维护：

```python
track_to_person: dict[int, str]
```

例如：

```text
Track 7  -> P001
Track 10 -> None
Track 14 -> P003
```

Track 消失后，可以删除临时绑定；Person ID 本身仍留在 Gallery。

ReID 找回后：

```text
old: Track 7 -> P001
new: Track 26 -> P001
```

---

## 19. 主循环伪代码

```python
while True:
    frame = video_source.read()
    if frame is None:
        break

    tracks = tracking_pipeline.process(frame)

    # 1. 更新当前目标是否仍存在
    target_manager.update_from_tracks(tracks)

    # 2. 判断哪些 Track 需要做 ReID
    reid_candidates = select_reid_candidates(
        tracks,
        target_manager,
        track_to_person,
        config,
    )

    crops = [crop_person(frame, t.bbox) for t in reid_candidates]
    embeddings = reid.extract_batch(crops) if crops else []

    for track, embedding in zip(reid_candidates, embeddings):
        # 当前目标 LOST 后优先尝试恢复
        if target_manager.is_lost:
            if target_manager.try_recover(track, embedding):
                continue

        # 未绑定永久身份的 Track 查询 Gallery
        if track.track_id not in track_to_person:
            match = gallery.search(embedding)
            if match.score >= config.reid.gallery_threshold:
                track_to_person[track.track_id] = match.person_id

    # 3. 用户框选
    if ui.has_roi_selection():
        roi = ui.get_roi()
        track = find_track_by_roi(roi, tracks)
        if track:
            crop = crop_person(frame, track.bbox)
            embedding = reid.extract(crop)
            target_manager.select_track(track, embedding)

    # 4. 用户入库
    if ui.enroll_clicked():
        person_id = target_manager.enroll_current_target(gallery)
        track_to_person[target_manager.selected_track_id] = person_id

    # 5. 高质量特征定期更新
    update_gallery_embeddings_if_needed(...)

    draw_result(...)
    ui.show(frame)
```

---

## 20. ReID 执行策略

ReID 不建议每帧对所有人运行。

推荐优先级：

### 高优先级

- LOST target 搜索；
- 新 Track；
- 用户刚框选的人。

### 中优先级

- 已识别 Person 的定期特征更新；
- 尚未匹配目标库、但需要周期重试的 Track。

### 不需要

- 已稳定跟踪且近期刚完成 ReID 的普通 Track 每一帧重复计算。

这能显著降低 GPU 负担。

---

## 21. UI 第一版

第一版直接用 OpenCV，优先保证功能完整。

MVP-3 在 OpenCV UI 中实现临时多目标选择；入库和 Gallery 管理仍属于后续 MVP。

MVP-3.1 将 `S`/`R` 实现为同一暂停画面上的连续鼠标编辑会话：进入会话时冻结
当前 frame 和 tracks，允许连续提交多个 ROI；Enter/Space 结束会话，Esc 取消
当前未完成拖框，Q 传播为程序退出。编辑会话使用主窗口 callback，不使用
`cv2.selectROI()` 或临时 ROI 窗口。

建议快捷键：

```text
S : Select Target / 框选并加入当前目标
R : Remove Target / 框选并移除一个目标
C : Clear Targets / 清除全部目标
Q : Quit
```

显示：

普通人：

```text
Track 12
```

当前临时目标：

```text
TARGET | ID 7
TARGET | ID 8
```

目标库命中：

```text
P001 | Track 26 | sim=0.86
```

当前目标且已入库：

```text
TARGET P001 | Track 26 | sim=0.86
```

后期竞赛展示需要更漂亮时，再做 PySide6。

---

## 22. 配置文件

`config/config.yaml` 建议：

```yaml
video:
  source: 0

model:
  yolo_weight: "weights/yolo/yolov8n.pt"
  device: "auto"
  person_class_id: 0
  conf_threshold: 0.35
  iou_threshold: 0.50
  image_size: 640

runtime:
  num_workers: 0

tracking:
  tracker: "config/trackers/botsort_baseline.yaml"
  persist: true
  show_track_id: true
  lost_frames: 15

selection:
  min_iou: 0.20

reid:
  model_name: "osnet_x0_25"
  weight: "weights/reid/osnet_x0_25_msmt17.pth"
  image_height: 256
  image_width: 128
  min_crop_width: 40
  min_crop_height: 100
  gallery_threshold: 0.70
  selected_target_threshold: 0.70
  max_embeddings_per_person: 50
  embedding_update_interval: 20
  min_bbox_width: 40
  min_bbox_height: 100

reid_recovery:
  lost_grace_frames: 5
  reference_update_interval_frames: 15
  recovery_interval_frames: 5
  max_reference_embeddings: 8
  recovery_threshold: 0.85
  recovery_margin: 0.05
  reference_update_threshold: 0.80
  recovery_reference_support_threshold: 0.80
  recovery_reference_support_top_k: 3
  recovery_min_track_age_frames: 3
  recovery_confirmation_hits: 2
  recovery_pending_max_age_frames: 60

reid_quality:
  min_track_confidence: 0.35
  max_edge_truncation_ratio: 0.30
  max_person_overlap_ratio: 0.60

gallery_recognition:
  enabled: true
  recognition_interval_frames: 10
  min_track_age_frames: 5
  recognition_threshold: 0.80
  recognition_margin: 0.05
  confirmation_hits: 2

gallery_enrichment:
  post_recovery_stable_frames: 30

ui:
  window_name: "Person Tracking - MVP-8.2"
  show_class_name: true
  show_confidence: true
  show_track_id: true
  show_unselected_tracks: false
  show_person_id: true
  show_similarity: true
  show_fps: true

database:
  path: "database/person_reid.db"

diagnostics:
  enabled: true
  log_interval_frames: 300
```

所有阈值和路径都放配置，不写死在业务代码中。

---

## 23. 日志与调试信息

建议使用 Python `logging`，而不是大量 `print()`。

MVP-1 至少记录：`APP_START`、`MODEL_LOADED`、`SOURCE_OPENED`、`SOURCE_END`、
`APP_STOP` 和异常信息；后续 MVP 再增加身份相关事件。

关键事件记录：

```text
TRACK_CREATED
TARGET_SELECTED
TARGET_LOST
TARGET_RECOVERED
PERSON_ENROLLED
GALLERY_MATCH
GALLERY_REJECT
PERSON_DELETED
DATABASE_LOADED
```

例如：

```text
[INFO] TARGET_SELECTED track=7
[INFO] TARGET_LOST old_track=7
[INFO] TARGET_RECOVERED old_track=7 new_track=26 score=0.84
[INFO] PERSON_ENROLLED person=P001 track=26
```

这对后期调错非常重要。

---

## 24. 性能策略

第一版先保证正确，再优化。

推荐顺序：

1. 使用较小 YOLO；
2. ReID batch inference；
3. 降低 ReID 调用频率；
4. embedding 全部预归一化；
5. Gallery 矩阵一次性向量乘法；
6. FP16；
7. 后期 ONNX/TensorRT。

如果目标库只有几十/几百人：

```python
scores = gallery_matrix @ query_embedding
```

即可，不需要 FAISS/向量数据库。

---

## 25. 典型问题和处理策略

### 25.1 同款衣服导致误识别

ReID 主要依赖外观，无法保证绝对身份准确。

处理：

- 多 embedding；
- 更严格阈值；
- 高质量 crop；
- OSNet 大模型；
- 竞赛演示场景合理控制。

### 25.2 人物换衣服

纯外观 ReID 很可能失败，这是能力边界。

### 25.3 多人交叉导致 ID Switch

先调 BoT-SORT；仍不够再引入 BoxMOT。

### 25.4 行人太小

设置最小 bbox 大小，小目标不进行永久身份判定。

### 25.5 Gallery 污染

只有**高置信 Person ID 绑定**时才能追加新 embedding。

绝不能：

```text
一次刚过阈值的模糊匹配 -> 立即把新 embedding 写入 Gallery
```

可以要求：

- 连续多次匹配一致；或
- 当前 Track 与 P001 已稳定绑定；或
- 当前人物是用户手动选中的目标。

### 25.6 一个 Track 同时匹配多个 Person

只接受最高分，并要求：

```text
best_score >= threshold
```

可选增强：

```text
best_score - second_best_score >= margin
```

降低近似人员误匹配。

---

## 26. MVP 开发阶段

### MVP-0：环境与项目骨架

- 创建目录；
- 配置文件；
- logging；
- 下载参考仓库；
- 确认 GPU/CPU 推理环境。

验收：项目能够启动并读取配置。

### MVP-1：YOLO 人员检测

```text
camera/video file -> YOLOv8n -> person bbox -> OpenCV display
```

验收：

- 支持摄像头索引和本地视频文件；
- 只保留 COCO `person` 类；
- 显示检测框、类别名和置信度；
- 自动选择 CUDA，CUDA 不可用时回退 CPU；
- Windows 默认 `num_workers=0`；
- 按 `q` 退出；
- 有基本 logging、输入/模型异常处理和最基本测试；
- 本阶段不实现 BoT-SORT、Track ID、OSNet/ReID、TargetGallery、SQLite 或复杂 UI。

### MVP-2：BoT-SORT

```text
YOLO -> BoT-SORT -> Track ID
```

验收：连续运动下 ID 基本稳定。

当前实现约束：主循环复用同一个 `TrackingPipeline`，模型只加载一次；每帧只调用
一次 `model.track()`，使用 `persist=True`、配置中的 BoT-SORT profile 和 `classes=[0]`。
MVP-2 不启用 BoT-SORT appearance ReID，不传 `workers`，不实现 ROI、OSNet/ReID、
TargetGallery、Person ID 或 SQLite。

### MVP-3：手动多目标选择

```text
ROI -> current Track -> selected_track_ids
```

验收：可重复框选多个当前行人；选中 Track 使用特殊框显示，`R` 按 ROI
移除单个目标，`C` 清除全部目标。Track 暂时消失时保留其 ID，Track ID
变化时不自动重新绑定。

当前实现约束：ROI 选择使用当前已经处理完成的帧和 `tracks` 列表，不重新运行
YOLO/BoT-SORT；MVP-3 不加载或调用 Torchreid/OSNet，不提取 embedding，
不创建 Person ID、TargetGallery 或 SQLite。

### MVP-3.1：暂停编辑会话

正常播放时默认只显示 selected Track；所有未选 Track 仍由 YOLO + BoT-SORT
后台跟踪。设置 `ui.show_unselected_tracks: true` 可恢复绿色调试框。
`S` 进入 ADD_TARGETS，`R` 进入 REMOVE_TARGETS；两个模式均使用当前帧的
frozen frame/frozen tracks 和主窗口 mouse callback，编辑期间不读取下一帧、
不调用 YOLO/BoT-SORT。Enter/Space 退出编辑，Q 退出程序，callback 必须在
正常和异常路径中清理。

### MVP-4：OSNet

只实现 Person ReID 特征提取和相似度验证：使用官方 Torchreid 1.4.0 的
`osnet_x0_25`，加载本地 MSMT17 combineall checkpoint，输出 normalized 512-D
`float32` embedding，并提供 cosine similarity 验证工具。MVP-4 不实现 LOST、
RECOVER、Track ID 重绑定、Person ID、TargetGallery、SQLite 或自动识别目标库。

验收：能够输出 shape 为 `(512,)` 或 `(N, 512)` 的 normalized embedding；在合理
测试素材上，同一人的相似度通常高于不同人的相似度，但不在本阶段固定最终阈值。

### MVP-5：LOST / RECOVER

当前 MVP-5 只实现当前进程内的 SessionTarget，不实现持久化 Person ID、历史
TargetGallery 或 SQLite。

用户选择 Track 时，先由 `crop_person` 提取合法人物 crop，再通过已加载一次的
Torchreid OSNet 建立 normalized 512-D reference。SessionTarget 至少保存：

```text
target_id
current_track_id
last_track_id
state = ACTIVE / LOST
reference_embeddings
normalized centroid
missing_frames
```

ACTIVE 目标按 `reference_update_interval_frames` 低频更新 reference，但新特征
必须达到 `reference_update_threshold` 的 centroid 一致性检查，并受
`max_reference_embeddings` 限制。current Track 连续缺失达到
  `lost_grace_frames` 后进入 LOST；恢复时只检查未被 ACTIVE target 占用的候选，按
  batch ReID 与 normalized centroid similarity 加 reference-support evidence 进行
  一对一匹配。匹配必须同时满足 `recovery_threshold` 与
  `recovery_reference_support_threshold`；reference support 是候选与 runtime
  reference bank 中 top-k 相似度的均值（reference 不足 k 条时使用全部）。存在
  second-best 时，target-side 和 candidate-side 都必须满足
  `best - second >= recovery_margin`，单候选/单目标的一侧自动通过 margin。任一
  绝对证据或 margin 不满足时保持 LOST；LOST 不要求最终必须恢复，允许无限期保持。
  Recovery 成功只重新绑定 Track，不立即把候选 embedding 加入 runtime reference bank。

验收：目标离开再返回，即使 Track ID 变了也恢复原 SessionTarget 的锁定；短暂
遮挡在 grace period 内不触发重绑定；错误候选或模糊匹配不强行恢复。

### MVP-6：内存 TargetGallery

MVP-6 只实现内存 `TargetGallery`，不实现 SQLite、磁盘持久化、程序重启恢复或
自动 Gallery recognition。用户必须先通过 `S` 创建 SessionTarget，再通过 `G`
显式 enrollment；普通 Track 不能直接入库。

`GalleryPerson` 使用递增的内存 `person_id`、默认 `Target P001` label、独立复制的
reference bank 和 normalized centroid。Gallery 内部维护
`session_target_id -> person_id` 关联，并负责其生命周期：R/C 删除或清除
SessionTarget 时解除关联但保留 GalleryPerson；删除 GalleryPerson 时清除所有
反向关联但保留 SessionTarget。target_id 和 person_id 在当前进程内均不复用。

G session 复用 MVP-3.1 的 frozen frame/tracks 和 mouse callback，可连续 enrollment
多个当前目标。enrollment 只复制 MVP-5 已有的 references/centroid，不重新运行
YOLO、BoT-SORT 或 OSNet。Gallery 不参与 MVP-5 recovery，也不对普通 Track 自动
执行 ReID。

验收：当前目标可获得 P001/P002 等 GalleryPerson；Track ID 变化不影响关联；重复
enrollment 幂等；删除 GalleryPerson、SessionTarget 或全部 SessionTargets 时边界
行为符合上述规则。

### MVP-7：SQLite 持久化

MVP-7 在不改变 MVP-6 `TargetGallery` 业务模型的前提下增加独立的
`GalleryRepository` SQLite 持久化层。SQLite 只保存 `GalleryPerson` 的
`person_id`、`label`、reference embeddings 和 normalized centroid；不保存
Track、SessionTarget、ACTIVE/LOST、current_track_id、missing_frames 或
session-target 映射。

embedding 使用固定的 float32、512-D BLOB 格式，不使用 pickle。每个 SQLite
connection 都启用 `PRAGMA foreign_keys = ON`，删除 GalleryPerson 时通过外键
级联删除其 embeddings。数据库父目录由 Repository 自动创建，默认相对路径为：

```yaml
database:
  path: "database/person_reid.db"
```

`gallery_meta.next_person_id` 只用于维护跨重启且删除后不复用的 person ID 序列。
新增 GalleryPerson、全部 embeddings 和序列更新在同一事务中提交，任一步失败都
整体回滚。启动时先初始化数据库、加载 GalleryPerson，再恢复内存 Gallery；历史
SessionTarget 映射不恢复。

G enrollment 仍然只能针对已经存在的 SessionTarget，且不重新运行 ReID。重复
enrollment 在当前进程内幂等，不会重复写入数据库。删除或清空 Gallery 时，内存
和磁盘由持久化协调层按明确顺序更新，失败不会静默制造不一致。

MVP-7 不实现 Gallery 自动识别。数据库加载出的 P001/P002 只存在于 Gallery，
不会自动绑定新的 SessionTarget，也不会对普通 Track 执行 ReID。`tools/gallery_admin.py`
提供离线的 `list`、`remove P001` 和 `clear` 管理操作；执行修改操作前应关闭主程序。

验收：关闭再启动后 Gallery 仍存在，person ID、label 和 512-D 特征保持不变；
新增人员继续使用下一个未分配 ID。

最终 Gallery UI 需求留待后续阶段实现：当前红框目标直接添加到库、已入库目标
从库中移除、独立 Gallery 侧栏/管理页、查看人员、单个删除、批量删除、清空和
label 编辑。

### MVP-8：自动目标库识别

MVP-8 在 MVP-7 启动加载的内存 `TargetGallery` 上增加独立的
`GalleryRecognitionCoordinator`。主循环顺序固定为：

```text
TrackingPipeline
    -> TargetRecoveryCoordinator（先处理已有 SessionTarget 的 LOST/RECOVER）
    -> GalleryRecognitionCoordinator
    -> visualization
```

Gallery recognition 只读取内存中的 `GalleryPerson`，不逐帧查询 SQLite，不调用
`TargetGallery.enroll()`，不创建新的 GalleryPerson，也不把新的 runtime feature
写回持久化 Gallery。只有当前没有 Gallery 绑定、有效且达到最小 Track 年龄的
person Track 才是候选；成功 Recovery、已被 ACTIVE SessionTarget 占用或其他明确
被占用的 Track 不参与。仅被 Recovery 检查但未成功认领的 Track 仍可参与，并可
复用同一 `frame_index` 的 `ReIDFrameCache` embedding。

识别采用 candidate embedding 与 Gallery normalized centroid 的 cosine similarity，
使用可配置 threshold、双侧 margin、确定性一对一匹配和 confirmation hits。单候选
或单目标一侧不存在 second-best 时，该侧 margin 自动通过；只有存在 second-best
才计算 `best - second >= margin`。pending 只在真正执行下一次 recognition 后结果
失败/身份变化，或 Track 消失时清除；识别间隔中的普通帧不清除 pending。缓存严格
限制在单个 frame index 内。

识别成功时，如果当前 Track 已由用户手动创建了未绑定 Gallery 的 SessionTarget，
就在该 SessionTarget 上 attach 现有 `person_id`；否则创建新的 SessionTarget，并
仅使用已经确认的当前 candidate embedding，调用与手动 S 相同的
`TargetManager.select()` 初始化单条 runtime reference bank。Persistent Gallery
references 只用于 cold-start 判定，不灌入新的 SessionTarget。
`TargetGallery.attach_session_target()` 严格保持 target/person 一对一，Track
ID 改变不影响关系。MVP-8 不自动更新 SQLite 中的 Gallery features。

默认初始工程参数为 `recognition_interval_frames=10`、`min_track_age_frames=5`、
`recognition_threshold=0.80`、`recognition_margin=0.05`、`confirmation_hits=2`；
这些值是保守的可调初值，不是通用最优值。

验收：程序重启后无需再次 S/G，已持久化的 P001/P002 在重新出现且通过保守匹配
后自动创建/绑定当前 SessionTarget，并显示 `TARGET P001 | ID n`。普通路人不应
被错误绑定；已识别目标后续仍由 MVP-5 Recovery 处理 Track 变化。

### MVP-8.1：拥挤场景与遮挡鲁棒性

本阶段优先处理两类不同现象：原 Track 消失并产生新 Track ID 的 Track
fragmentation，以及 Track fragmentation 后 Recovery 将相似衣着路人错误绑定的
false recovery。没有人工 ground truth 时，只记录 `TRACK_CREATED`、`TRACK_ENDED`、
`TARGET_LOST` 和 `TARGET_RECOVERED old_track_id -> new_track_id`，不把任意 Track
ID 变化自动命名为确定的 fragmentation，也不自动判断 Recovery 的真假。

Recovery 增加连续 Track age、可配置的 ReID crop quality gate，以及同一
`target_id + candidate_track_id` 的多次有效确认；第一次达标只进入 pending，默认
两次真正的、高质量且通过 threshold 和双侧 margin 的 Recovery attempt 后才恢复。
质量拒绝不是 identity mismatch，不清除 pending；candidate 消失、后续有效 attempt
失败或候选身份改变时才清除，并通过最大 pending 寿命避免 stale 状态。quality gate
至少检查 Track confidence、crop 大小、frame-edge 截断和
`intersection(target, other_person) / area(target)` 的最大 person overlap ratio。
不合格 crop 仍可用于 tracker，但不能用于 reference update、Recovery confirmation
或 Gallery recognition confirmation。

Recovery 先于 Gallery recognition 执行。Recovery 只成功占用的 Track 才会传给后者
排除；仅被 Recovery 检查但没有成功认领的 Track 仍可参加 Gallery recognition，并
复用同一帧的 `ReIDFrameCache`，缓存不跨帧复用。MVP-8.1 暂不把 ActiveIdentityGuard
接入主链，避免在尚未观察到 Track ID 不变但人物切换时增加每帧 OSNet 开销或误触发
LOST。

提供 `config/trackers/` 下的 baseline、crowd 和可选 appearance-ReID BoT-SORT
配置。检测 conf、`track_buffer`、`imgsz` 和 `with_reid` 按固定 crowd regression
video、固定目标和固定时间段做单变量 A/B；不直接宣称某组参数最优。运行诊断输出
processed frames、Track 创建/结束、TARGET LOST/RECOVERED、Recovery attempted/
pending/accepted、quality rejected、Gallery recognized 和平均 FPS。false recovery
由人工结合视频标记，评价优先级为 false recovery > temporary LOST > recovery 速度。

默认工程初值为 `recovery_min_track_age_frames=3`、
`recovery_confirmation_hits=2`、`recovery_pending_max_age_frames=60`、
`min_track_confidence=0.35`、`max_edge_truncation_ratio=0.30` 和
`max_person_overlap_ratio=0.60`；这些均为可调的工程起点，不是通用最优值。

### MVP-8.2：安全的持久化 Gallery 特征增量

MVP-8.2 只实现安全的 Gallery feature enrichment，不改变 SQLite schema，也不
增加后台写库线程。用户在当前运行中通过 `S` 选择目标并通过 `G` 显式入库时，
GalleryPerson 立即创建，即使当时只有一份有效 reference。之后只有该
SessionTarget 在本次运行中明确 G enrollment 后，被 TargetManager 接受的、
已经通过 MVP-5/MVP-8.1 quality gate 和 centroid 一致性检查的新 reference，才
可以用于更新对应的 GalleryPerson。

reference 更新由已有 ReID 流程产生，不为 Gallery enrichment 额外运行 OSNet。
`TargetRecoveryCoordinator` 在 `add_reference()` 成功后产生一次性的
`ReferenceUpdateEvent`；主循环通过 drain API 消费，旧事件不会在后续帧重复写库。
被拒绝的 reference、LOST 期间、recovery pending 期间和 recovery candidate
不会更新持久化 Gallery。自动识别得到的 SessionTarget 默认没有本次运行的
enrichment 资格；用户随后按 `G` 时，只为已有的 P001 等目标授予资格，不创建
新的 GalleryPerson。

SessionTarget 的 reference bank 继续服务短期 LOST/Recovery，保持最多 8 条的
FIFO 运行期策略；持久化 Gallery 不直接镜像这组 runtime references。每次只使用
`ReferenceUpdateEvent.accepted_embedding` 作为增量候选，在当前持久化 bank 上维护
最多 8 条代表性 reference：入库时的 index 0 是受保护 anchor，候选与任一已有
reference 的 cosine 达到 `duplicate_similarity_threshold` 时丢弃；bank 已满且
候选具有新信息时，只能替换最冗余的非 anchor reference，且候选冗余必须严格更低。
bank 真正变化后重新计算 normalized centroid，并构造 deep-copied feature snapshot。
Repository 先在同一 SQLite transaction 中替换该 person 的 centroid 和 embedding
rows；事务成功后才将同一份已校验 snapshot 应用到内存 TargetGallery。数据库失败
时保留旧的内存和磁盘版本；若内存应用发生异常则记录 consistency error，并从
Repository 重新加载该 person 恢复一致性。`person_id`、label、session-target
mapping 和 `next_person_id` 均不改变。

默认工程参数为 `gallery_enrichment.max_reference_embeddings=8` 和
`gallery_enrichment.duplicate_similarity_threshold=0.97`；它们是可调工程初值，
不声明为通用最优值。

目标从 LOST 恢复后，必须连续 ACTIVE 达到配置的
`gallery_enrichment.post_recovery_stable_frames`（初始工程值 30）才恢复 enrichment；
再次 LOST 会重新阻止。该 cooldown 只影响 recovery 后的增量更新，不影响首次
S->G 入库。所有数值均为可调工程初值，不声明为通用最优值。

验收：S 后立即 G 可以创建 P001；目标继续稳定活动并接受新 reference 后，P001 的
embedding 数量逐步增加且不超过持久化 bank 的最大值；关闭遮挡、LOST、recovery pending
或低质量期间不增加；重启后自动识别 P001 仍不会自动更新其持久化特征，除非本次
运行用户再次明确 G。

### Atlas 310B 部署适配（横切部署变体）

Atlas 适配不是新的业务 MVP，也不改变 MVP-8.2 的身份、Recovery、Gallery、SQLite
或 OpenCV UI 规则。项目保留双 inference backend：PC 默认使用
`inference.backend: torch`；Atlas 使用 `inference.backend: ascend`。Atlas 只替换
YOLO 与 OSNet 的神经网络推理后端：

```text
视频帧 -> YOLO OM / pyACL -> CPU BoT-SORT -> Track[] -> 既有业务逻辑
Track crop -> OSNet OM / pyACL -> normalized 512-D embedding
```

Atlas 路径禁止 `torch_npu`、`YOLO.predict()`、`YOLO.track()`、Torchreid forward
和 CUDA tensor。Ultralytics 8.4.138 仅作为 CPU BoT-SORT 实现使用，BoT-SORT
appearance ReID 默认关闭。`TrackingPipeline.process(frame)` 与
`ReIDExtractor.extract/extract_batch` 的上层接口保持不变；业务层不感知 Torch
还是 Ascend 后端。

PC 导出脚本固定 ONNX opset 11。YOLO 使用当前配置的模型权重、静态
`1x3x640x640`、batch=1、`dynamic=false`、`simplify=false`、`nms=false`，输入名为
`images`，输出保留 raw detection。OSNet 使用官方 MSMT17 checkpoint 的 feature
输出而非 classifier logits，输入为 `Nx3x256x128`，输出为 `Nx512`，输入/输出名为
`images`/`embedding`。导出后必须检查 opset、输入名、维度和输出结构；CANN 6.2.RC2
转换脚本默认 `SOC_VERSION=Ascend310B4`，但允许通过环境变量覆盖，不在代码中永久
写死 SoC。

`src/ascend_runtime.py` 统一管理 `acl.init`、device/context、OM load/execute 和
逆序释放；YOLO 与 OSNet 共享一个运行时，模型只加载一次。OSNet 使用配置的动态
batch `[1,2,4,8]`，超出时按真实 crop 分块，不补假样本。YOLO preprocessing、
letterbox 坐标映射、raw output decode、person filter 和 CPU NMS 由 Ascend 适配层
完成。现有 `ReIDFrameCache` 仍只允许同一帧复用 embedding，不跨帧复用。

部署文件位于 `deploy/atlas/`，完整命令、Atlas 依赖、smoke test 和 benchmark 见
`deploy/atlas/README.md`。默认 Atlas 配置为 `config/config_atlas.yaml`，模型路径
使用项目相对路径 `weights/atlas/*.om`。Atlas 硬件上的 `import acl`、OM 执行、
`npu-smi info` 和 AICore 非零使用率必须单独验收；PC 单元测试不得声称完成这些
硬件验证。SQLite schema、TargetGallery、SessionTarget、Recovery、Gallery
recognition、Gallery enrichment 和 UI 行为均保持不变。

### MVP-9：真实 RTSP

验收：本地测试视频/摄像头之外，稳定切换到真实 RTSP 实时流。

### MVP-10：竞赛展示优化

- FPS；
- UI；
- 目标列表；
- 操作提示；
- 异常处理；
- Demo 视频。

---

## 27. 验收测试场景

### Case A：连续跟踪

- 三个人同时出现；
- 用户选择 B；
- B 行走；
- 目标框持续跟随 B。

### Case B：短时遮挡

- B 被其他人挡住几帧；
- 重新出现后仍持续锁定。

### Case C：完全离开再返回

- B 离开画面；
- 原 Track ID 失效；
- B 再回来；
- 系统通过 ReID 自动恢复。

### Case D：目标入库

- B 加入 P001；
- 程序退出；
- 程序重启；
- B 再出现；
- 自动显示 P001。

### Case E：普通路人不过度误报

- P001 不在场；
- 多个普通路人经过；
- 不应频繁错误显示 P001。

### Case F：多人交叉

- 两人交叉；
- Tracker 若短暂 ID Switch，Person ID 后续应尽量被 ReID 校正。

### Case G：同一目标不同方向

- 正面注册；
- 侧面/背面出现；
- 通过多 embedding 逐渐提高重识别稳定性。

---

## 28. 测试策略

### 单元测试

优先测试不依赖 GPU 的业务逻辑：

- ROI 与 bbox IoU；
- Person ID 生成；
- Gallery add/search/delete；
- centroid；
- SQLite 保存/加载；
- TargetManager 状态转移。

### 集成测试

- 一个短视频验证 tracking；
- 一个“离开再回来”视频验证 ReID；
- 一个多人视频验证误识别。

Codex 每完成一个 MVP，应至少补对应测试或最小可运行脚本。

---

## 29. Codex 协作规则

主项目根目录同时放置 `AGENTS.md`。

Codex 应遵守：

1. 以本设计文档为需求依据；
2. 不擅自增加研究型复杂度；
3. 不一次性重写整个工程；
4. 按 MVP 顺序逐步实现；
5. `references/` 只读；
6. 优先使用成熟库；
7. `Track ID` 与 `Person ID` 必须分离；
8. 所有路人允许 Track，但只有用户主动入库者拥有永久 Person ID；
9. 所有阈值配置化；
10. 每个模块保持清晰接口；
11. 优先小改动、可运行、可测试；
12. 如果需要改变架构，先说明原因，不自行大规模改写。

---

## 30. 推荐 requirements（按 MVP 分阶段）

MVP-1 的精简依赖按当前已验证环境锁定为：

```text
ultralytics==8.4.138
torch==1.13.1
torchvision==0.14.1
opencv-python==4.10.0.84
numpy==1.26.4
PyYAML==6.0.3
```

MVP-4 接入 ReID 时使用官方
`https://github.com/KaiyangZhou/deep-person-reid` 项目提供的 Torchreid 1.4.0
实现/API；不能把 PyPI 上名称相同但来源和版本不确定的 `torchreid` 包作为正式
依赖。正式 requirements 使用固定的官方 Git commit，不写成普通
`torchreid==1.4.0`。MVP-3.1 及之前不安装、不加载或调用 Torchreid/OSNet。
Pillow 由实际依赖链提供。

正式 requirements 中的固定条目为：

```text
torchreid @ git+https://github.com/KaiyangZhou/deep-person-reid.git@f8cd150fdf77e8d9e1ed143b7f308c2c609ded50
```

SQLite 使用 Python 标准库 `sqlite3`，无需额外 pip 安装。

如果后期使用 PySide6：

```text
PySide6
```

BoxMOT 不应在第一版 requirements 中默认加入，除非实际决定切换 tracker backend。

---

## 31. 当前最终推荐技术路线

```text
                 Ultralytics
              YOLO + BoT-SORT
                      │
                      ▼
               temporary Track ID
                      │
                      ▼
                Torchreid OSNet
                      │
                      ▼
                ReID embedding
                      │
                      ▼
             自研 TargetManager
                      │
                      ▼
              自研 TargetGallery
                      │
                      ▼
                   SQLite
                      │
                      ▼
              persistent Person ID
```

第三方项目职责：

```text
Ultralytics          -> 检测 + 第一版跟踪
deep-person-reid     -> OSNet
Re-id_initial        -> 小型 Gallery/集成参考
person-reid-yolov8-  -> 多 embedding / centroid / re-entry 参考
tracking
person_reid_yolo     -> SQLite/持久化参考
BoxMOT               -> Tracker 增强备用
```

因此第一版不是把六个 GitHub 项目全部拼在一起，而是：

> **两个成熟核心组件负责算法，三个小项目负责业务实现参考，一个大框架作为 tracker 备用。**

---

## 32. 后续可选增强

完成所有 MVP 后再考虑：

- PySide6 图形界面；
- 多目标同时重点关注；
- 目标库管理页面；
- 目标出现告警；
- 历史出现时间记录；
- 多摄像头跨镜 ReID；
- 人脸 + ReID 多模态融合；
- FAISS；
- ONNX / TensorRT；
- 自定义 ReID 微调；
- 轨迹回放；
- 目标截图/录像自动保存。

---

## 33. 最终一句话定义

本项目可以定义为：

> **一个基于 YOLO 多目标检测跟踪和 Person ReID 的交互式人员目标锁定系统：用户可从实时视频中手动选择目标，系统持续跟踪；目标离开后依靠 ReID 重新识别；用户可将目标加入本地 Gallery，使其在未来再次出现时自动被识别和标记。**

第一版应始终围绕“功能完整、稳定演示、代码简单、便于 Codex 逐步实现”进行开发，而不是追求算法创新。
