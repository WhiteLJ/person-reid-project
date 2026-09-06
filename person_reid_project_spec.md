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

## 3. 两类身份必须严格区分

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
MVP-3 只保存临时 `Track ID`，不提取 crop、不运行 OSNet/ReID，也不创建 Person ID。

---

### 9.3 当前目标 TRACKING

如果某个 `selected_track_id` 仍然存在：

- 直接跟随该 Track；
- 绘制特殊颜色框；
- 当前 Track 暂时消失时保留其 Track ID；
- 如果 BoT-SORT 恢复相同 Track ID，则继续高亮；
- MVP-3 不进行 embedding 或全库搜索。

---

### 9.4 当前目标 LOST

当连续若干帧没有找到 `selected_track_id`：

```text
TRACKING -> LOST
```

保留：

```text
reference_embeddings
selected_person_id（若已入库）
last_seen_frame
last_seen_timestamp
last_bbox
```

然后进入 ReID Search。

---

### 9.5 目标重新进入

对于新 Track：

```text
new track
 ↓
crop
 ↓
OSNet
 ↓
query embedding
 ↓
与 selected target references 比较
```

达到阈值：

```text
new_track_id -> 当前 Target
LOST -> TRACKING
```

如果当前目标已是 P001：

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

```text
NONE
 │
 │ 用户框选
 ▼
TRACKING
 │
 │ 连续 N 帧消失
 ▼
LOST
 │
 │ ReID match
 └──────────────> TRACKING

用户取消：任意状态 -> NONE
```

如果只是临时锁定：

```text
state = TRACKING
person_id = None
```

如果已经入库：

```text
state = TRACKING
person_id = P001
```

推荐字段：

```text
state
selected_track_id
selected_person_id
reference_embeddings
last_seen_frame
last_seen_timestamp
last_bbox
lost_frame_count
```

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
    def select_track(self, track, embedding): ...
    def update_visible(self, track): ...
    def mark_lost(self): ...
    def try_recover(self, track, embedding) -> bool: ...
    def enroll_current_target(self, gallery) -> str: ...
    def cancel(self): ...
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
  tracker: "botsort.yaml"
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

ui:
  window_name: "Person Tracking - MVP-3"
  show_class_name: true
  show_confidence: true
  show_track_id: true
  show_unselected_tracks: false
  show_person_id: true
  show_similarity: true
  show_fps: true

storage:
  database: "data/gallery.db"
  thumbnails_dir: "data/thumbnails"
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
一次 `model.track()`，使用 `persist=True`、`tracker="botsort.yaml"` 和 `classes=[0]`。
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

验收：目标离开再返回，即使 Track ID 变了也恢复锁定。

### MVP-6：内存 TargetGallery

验收：当前目标可获得 P001 并在当前运行期重新识别。

### MVP-7：SQLite

验收：关闭再启动后 Gallery 仍存在。

### MVP-8：自动目标库识别

验收：目标库人员无需再次手动框选即可自动显示 Person ID。

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
