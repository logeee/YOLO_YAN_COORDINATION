# YOLO 上表面检测模块说明

日期：2026-05-27

这个文档只说明 YOLO 检测层。它和 PnP/距离/坐标计算已经拆开：

```text
YOLO 模块：输入一张图，输出一个或多个烟盒上表面 mask 候选和四点
坐标模块：从 YOLO 候选里选一个，用这个候选的四点计算 PnP/XYZ
```

## 文件位置

YOLO 独立模块：

```bash
scripts/yolo_topface_detector.py
```

坐标 API：

```bash
scripts/cigarette_pose_optical_api.py
```

常驻服务：

```bash
scripts/cigarette_pose_yolo_server.py
scripts/cigarette_pose_yolo_server_gpu.sh
```

模型：

```bash
models/YanHe20class.pt
```

2026-07-01 默认模型已换成 `models/YanHe20class.pt`，用于 20 类烟盒 segmentation。模型类别名为数字编码：`31019915`、`43010159`、`48013265`、`33013189`、`42013109`、`33013181`、`42013086`、`42013085`、`34063141`、`34063140`、`34063147`、`42013097`、`42013096`、`34063142`、`42013075`、`34063136`、`34063135`、`48090225`、`48090217`、`48090201`。YOLO 检测层会保留模型返回的所有类别候选，并在每个 candidate 里返回 `class_id` / `class_name`，不在检测层做类别过滤。

坐标层会按左目选中 candidate 的类别选择物理尺寸：

```text
XiongMao    = 161mm x 95mm
Xizi_Liqun  = 280mm x 89mm
```

注意：这个尺寸切换发生在 `cigarette_pose_optical_api.py` 的 PnP/XYZ 层；本模块仍只负责 YOLO mask、类别、置信度和四点输出。

## 解耦关系

YOLO 模块不计算距离，也不计算相机坐标。它只做这些事：

```text
1. 调用 Ultralytics YOLO segmentation
2. 保留 YOLO 返回的所有 mask
3. 每个 mask 拟合出上表面四点 points_px
4. 给候选排序
5. 返回所有候选 candidates
6. 同时返回一个 selected 候选，供 PnP 层使用
```

坐标层只依赖：

```text
selected.points_px
```

所以以后如果要换模型，只要新模块能输出同样格式的四点候选，PnP/XYZ 这层不用动。

## Python 调用

```python
from yolo_topface_detector import detect_yolo_points_from_image

detection, info = detect_yolo_points_from_image(
    image,
    model_path="models/YanHe20class.pt",
    conf=0.15,
    imgsz=640,
    device="cuda:0",
    mask_threshold=0.5,
    select="score",
    select_index=0,
)

selected_points = detection.points
all_candidates = info["candidates"]
```

返回的 `detection` 是被选中用于 PnP 的单个候选；`info["candidates"]` 是 YOLO 返回的所有候选。

## 单独测试 YOLO

只跑 YOLO，不算坐标：

```bash
cd ~/unifolm-world-model-action/robot_client_unitree_g1_full_20260509/repos/unitree_deploy
source /home/unitree/venvs/tv_gpu/bin/activate

PYTHONPATH="$PWD:$PWD/scripts" python scripts/yolo_topface_detector.py \
  --image /tmp/left_input.jpg \
  --model models/YanHe20class.pt \
  --device cuda:0 \
  --pretty
```

这会输出：

```text
selected      当前选中的候选
candidates    所有 YOLO mask 候选
```

## 候选字段

每个 candidate 包含：

```text
candidate_index     排序后的候选序号
raw_yolo_index      YOLO 原始返回序号
class_id            类别 ID
class_name          类别名，例如 XiongMao
confidence          YOLO 置信度
score               当前排序分数
box_xyxy            YOLO 检测框
box_center_px       检测框中心
mask_area_px        mask 面积
points_px           由 mask 拟合出的四个上表面点
device              推理设备，GPU 应为 cuda:0
```

`points_px` 顺序：

```text
0 = 左上
1 = 右上
2 = 右下
3 = 左下
```

## 多目标选择

YOLO 可以返回多个 mask，但 PnP/XYZ 一次只用一个。选择参数：

```text
select       候选排序规则
select_index 排序后选第几个候选
```

可用排序规则：

```text
score       默认，主要按 confidence，mask 面积只做很小的辅助加分
largest     mask 面积最大
leftmost    图像里最左
rightmost   图像里最右
topmost     图像里最上
bottommost  图像里最下
center      最靠近图像中心
index       YOLO 原始返回顺序
```

例子：按 score 选第 2 个目标：

```bash
curl -s "http://127.0.0.1:18081/xyz?yolo_select=score&yolo_index=1"
```

默认 `/xyz` 等价于：

```bash
curl -s "http://127.0.0.1:18081/xyz?yolo_select=score&yolo_index=0"
```

也就是按 `score` 排序后选择第 0 个候选。

例子：选最左边目标：

```bash
curl -s "http://127.0.0.1:18081/xyz?yolo_select=leftmost&yolo_index=0"
```

## 调试页面

常驻服务的一页看全：

```text
http://192.168.0.149:18081/debug
```

如果网络不方便直接访问，也可以本地 SSH 转发看 `215`：

```powershell
ssh -L 18082:127.0.0.1:18081 unitree@192.168.0.215
```

浏览器打开：

```text
http://127.0.0.1:18082/debug
```

看所有候选叠加图：

```text
http://127.0.0.1:18082/debug/left_candidates.jpg
```

图上标签格式：

```text
* #0 XiongMao conf=0.954
```

其中 `*` 表示当前被选中用于 PnP/XYZ 的候选。

## 当前边界

YOLO 模块负责：

```text
图像 -> 多个 mask 候选 -> 每个候选四点
```

坐标模块负责：

```text
一个候选四点 -> PnP -> center_xyz_mm / range_from_left_camera_mm
```

两层之间唯一需要长期稳定的接口是：

```text
candidates[*].points_px
selected.points_px
```
## 标签过滤

YOLO 检测层支持按类别标签过滤候选。传入标签后，只保留匹配该标签的 mask，再按 `confidence` 选择置信度最高的候选。

命令行示例：

```bash
PYTHONPATH="$PWD:$PWD/scripts" python scripts/yolo_topface_detector.py \
  --image /tmp/left.jpg \
  --model models/YanHe20class.pt \
  --label XiongMao \
  --select confidence \
  --pretty
```

坐标服务示例：

```bash
curl -s "http://127.0.0.1:18081/xyz?label=XiongMao"
curl -s "http://127.0.0.1:18081/xyz?label=Liqun"
```

`Liqun` 会匹配模型类别名 `Xizi_Liqun`。候选和选中结果里都会保留 `class_id`、`class_name`、`confidence`，便于下游确认当前坐标来自哪一类烟盒。
