# 香烟盒左相机 Optical 坐标 API

日期：2026-05-25

这个接口用于输出香烟盒上表面中心点在**左相机 optical 坐标系**下的位置。脚本只做感知和坐标计算，不控制机器人。

## 文件位置

机器人本体脚本：

```bash
/home/unitree/unifolm-world-model-action/robot_client_unitree_g1_full_20260509/repos/unitree_deploy/scripts/cigarette_pose_optical_api.py
```

独立 YOLO 检测模块：

```bash
/home/unitree/unifolm-world-model-action/robot_client_unitree_g1_full_20260509/repos/unitree_deploy/scripts/yolo_topface_detector.py
```

YOLO 模块单独说明：

```bash
/home/unitree/unifolm-world-model-action/robot_client_unitree_g1_full_20260509/repos/unitree_deploy/docs/yolo_topface_detector_module.md
```

依赖的检测/PnP 模块：

```bash
/home/unitree/unifolm-world-model-action/robot_client_unitree_g1_full_20260509/repos/unitree_deploy/scripts/auto_pnp_cuboid_depth.py
```

本文档位置：

```bash
/home/unitree/unifolm-world-model-action/robot_client_unitree_g1_full_20260509/repos/unitree_deploy/docs/cigarette_pose_optical_api.md
```

## 坐标系定义

输出坐标系：

```text
left_camera_optical
```

轴方向：

```text
+X = 图像右方
+Y = 图像下方
+Z = 相机前方，也就是光轴深度方向
单位 = mm
```

返回的 `center_xyz_mm` 是香烟盒**上表面矩形中心点**在左相机 optical 坐标系下的坐标，不是机器人 base 坐标。

这里有三个容易混淆的量：

```text
z_mm / left_depth_mm / optical_axis_depth_mm
  = 沿左相机 optical Z 轴的深度，也就是光轴方向深度。

range_from_left_camera_mm
  = 左相机光心到目标中心点的直线距离，计算方式是 sqrt(x_mm^2 + y_mm^2 + z_mm^2)。

机器人 base 坐标里的前后/左右/上下
  = 需要左相机到机器人 base 的外参，也就是相机安装角度和平移。当前脚本不输出 base 坐标。
```

注意：`cv2.solvePnP` 原始输出是 OpenCV 相机坐标：

```text
X_cv = 图像右方
Y_cv = 图像下方
Z_cv = 相机前方/深度
```

本 API 会把它转换成项目使用的 optical 坐标：

```text
X = X_cv
Y = Y_cv
Z = Z_cv
```

也就是说，当前项目里的 `left_camera_optical` 和 OpenCV 相机坐标方向一致：右、下、前分别是正 X、正 Y、正 Z。

## 直接运行

在机器人上执行：

```bash
cd ~/unifolm-world-model-action/robot_client_unitree_g1_full_20260509/repos/unitree_deploy

bash scripts/cigarette_pose_yolo_gpu.sh
```

如果下游只需要 `[x, y, z]`，可以直接用：

```bash
cd ~/unifolm-world-model-action/robot_client_unitree_g1_full_20260509/repos/unitree_deploy

bash scripts/cigarette_pose_yolo_gpu.sh | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['center_xyz_mm'])"
```

只打印中心点垂直地面上方 10cm 的点：

```bash
curl -s http://127.0.0.1:18081/xyz | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['center_above_xyz_mm'])"
```

如果临时要改中心点 above 高度，例如 80mm：

```bash
curl -s "http://127.0.0.1:18081/xyz?center_above_height_mm=80" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['center_above_xyz_mm'])"
```

如果下游需要直线距离和坐标，建议读取完整 JSON 里的这两个字段：

```text
center_xyz_mm                 [x_mm, y_mm, z_mm]
center_above_xyz_mm        中心点沿地面垂直向上 100mm 后的位置
range_from_left_camera_mm     左相机光心到上表面中心点的直线距离
```

上面这个命令会进入专门给 Jetson GPU 建好的环境：

```bash
/home/unitree/venvs/tv_gpu
```

并且显式使用：

```bash
--mode yolo
--yolo-model models/Liqun_Xiongmao.pt
--yolo-device cuda:0
```

输出 JSON 里可以用这个字段确认是否真的走 GPU：

```text
point_adjustments.left_yolo.device = cuda:0
point_adjustments.right_yolo.device = cuda:0
```

注意：如果每次都从命令行启动脚本，耗时会包含 Python 启动、模型加载和拍照时间；真正的 YOLO GPU 热推理在同一个进程里会快很多。机器人程序如果要高频调用，建议 import 这个脚本里的 API，让进程常驻，这样 YOLO 模型会被缓存复用。

旧的 conda `tv` 环境可以跑 CPU/调试，但不建议作为机器人本体的默认调用方式：

```bash
cd ~/unifolm-world-model-action/robot_client_unitree_g1_full_20260509/repos/unitree_deploy

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tv

PYTHONPATH="$PWD:$PWD/scripts" python3 scripts/cigarette_pose_optical_api.py \
  --capture \
  --pretty
```

脚本会打印一个 JSON，同时在 `/tmp/cigarette_pose_optical_YYYYMMDD_HHMMSS/` 下保存输入图、点图和结果 JSON。

默认流程**不需要外部提供真实直线距离**。脚本会用当前相机参数、香烟盒上表面尺寸和自动检测到的四个点，通过 PnP 直接估计：

```text
center_xyz_mm                 左相机 optical 坐标
range_from_left_camera_mm     左相机到中心点的估计直线距离
z_mm / left_depth_mm          光轴方向深度，不是直线距离
```

## 主要输出字段

示例：

```json
{
  "ok": true,
  "frame": "left_camera_optical",
  "center_xyz_mm": [109.3, 266.9, 739.2],
  "x_mm": 109.3,
  "y_mm": 266.9,
  "z_mm": 739.2,
  "left_depth_mm": 739.2,
  "optical_axis_depth_mm": 739.2,
  "range_from_left_camera_mm": 793.5,
  "direction_unit_xyz": [0.1377, 0.3364, 0.9316],
  "coordinate_method": "range_times_direction",
  "opencv_camera_xyz_mm": [109.3, 266.9, 739.2],
  "points_px": [[341, 312], [373, 315], [381, 364], [344, 364]],
  "selected_orientation": "short_x_long_y"
}
```

下游优先使用这些字段：

```text
center_xyz_mm
x_mm
y_mm
z_mm
left_depth_mm
optical_axis_depth_mm
range_from_left_camera_mm
direction_unit_xyz
opencv_camera_xyz_mm
```

调试和质量检查字段：

```text
points_px                  左图上表面四个角点
left_reprojection_error_px 左图 PnP 重投影误差
right_depth_mm             右图深度，仅作参考
depth_delta_mm             abs(left_depth_mm - right_depth_mm)
range_from_left_camera_mm  左相机光心到目标中心点的直线距离
direction_unit_xyz         从左相机指向目标中心点的单位方向向量
stereo_check               用 60mm 左右镜头表面距离做的双目 disparity 诊断
debug_images.left_points   左图四点可视化
debug_images.left_points_raw 左图 mask 原始四点可视化
debug_images.right_points  右图四点可视化
point_adjustments          mask 原始四点到自动修正四点的记录
warnings                   ROI 或一致性警告
```

当前策略是：**左相机结果为主，右相机只做一致性检查**。

下游如果要用图里的坐标轴，使用 `center_xyz_mm` / `x_mm` / `y_mm` / `z_mm`。这些坐标由下面这个关系得到：

```text
center_xyz_mm = range_from_left_camera_mm * direction_unit_xyz
```

如果要左镜头到目标中心点的直线距离，使用 `range_from_left_camera_mm`。`opencv_camera_xyz_mm` 只用于调试和核对旧结果。

## 可选：已知真实直线距离时

这一节只用于校准、验算或强制按实测距离缩放坐标；正常自动化调用不要加 `--known-range-mm`。

如果现场已经量到了“左目镜头玻璃表面到香烟盒上表面中心点”的真实直线距离，可以把这个距离传给脚本：

```bash
PYTHONPATH="$PWD:$PWD/scripts" python3 scripts/cigarette_pose_optical_api.py \
  --capture \
  --known-range-mm 700 \
  --pretty
```

如果这组数据要用来反推 focal，同时加：

```bash
PYTHONPATH="$PWD:$PWD/scripts" python3 scripts/cigarette_pose_optical_api.py \
  --capture \
  --known-range-mm 600 \
  --calibrate-focal-from-known-range \
  --pretty
```

输出里的 `focal_calibration.focal_px` 就是“这一组样本建议的 focal”。建议收集 3 组以上，再取稳定值更新默认参数。

这时脚本会：

```text
1. 先用 mask 或 YOLO 四点算出相机坐标方向。
2. 再把坐标向量缩放到 --known-range-mm 指定的真实直线距离。
3. 输出缩放后的 center_xyz_mm / x_mm / y_mm / z_mm。
```

输出里会多一个字段：

```text
range_override
```

里面会记录缩放前的 PnP 坐标和缩放比例，例如：

```json
{
  "range_override": {
    "enabled": true,
    "known_range_from_left_lens_glass_mm": 700.0,
    "lens_glass_to_optical_center_mm": 0.0,
    "applied_range_from_left_optical_center_mm": 700.0,
    "scale": 1.073952,
    "pnp_unscaled_center_xyz_mm": [115.8, 234.4, 597.1],
    "pnp_unscaled_range_from_left_camera_mm": 651.8
  }
}
```

默认假设左目镜头玻璃表面和 optical center 的偏移为 0。如果之后知道 optical center 在玻璃后方的偏移量，可以加：

```bash
--lens-glass-to-optical-center-mm 5
```

这个偏移通常只有几毫米，远小于 600-800mm 的工作距离；不知道时先保持默认 0。

## 四个角点顺序

脚本内部使用的四点顺序是：

```text
0 = 左上角
1 = 右上角
2 = 右下角
3 = 左下角
```

这里的“上/下”指的是图像坐标里的上/下，不是机器人世界坐标里的上/下。

默认命令行参数是 `--points-order auto`，所以如果之后 YOLO/keypoint 模型输出的是无序四点，脚本会自动按图像位置排序。如果模型已经按上面顺序输出，可以使用 `--points-order ordered`。

## 当前 YOLO 模式

默认模式已经切到 `yolo`，使用 `models/Liqun_Xiongmao.pt` 的 segmentation mask 直接拟合香烟盒上表面四点，不再走旧的颜色/暗色 mask 检测：

```bash
PYTHONPATH="$PWD:$PWD/scripts" python3 scripts/cigarette_pose_optical_api.py \
  --capture \
  --pretty
```

当前默认参数：

```text
long-side-m  = 0.161
short-side-m = 0.095
mode         = yolo
yolo-model   = models/Liqun_Xiongmao.pt
yolo-conf    = 0.15
yolo-imgsz   = 640
yolo-device  = auto
orientation  = auto_by_stereo
focal-px     = 260.0
cx           = 320.0
cy           = 240.0
stereo-baseline-mm = 60.0
```

2026-05-27 起默认模型已更新为 `models/Liqun_Xiongmao.pt`，用于识别利群和熊猫两种烟盒。模型类别名为 `Xizi_Liqun` 和 `XiongMao`。接口默认不按类别过滤；YOLO 返回的每个候选都会带 `class_id` / `class_name`，坐标计算仍按 `yolo_select` 和 `yolo_index` 从所有候选里选一个。

YOLO 模式会按左目选中候选的类别自动切换上表面尺寸再计算 PnP/XYZ：

```text
XiongMao    = 161mm x 95mm
Xizi_Liqun  = 280mm x 89mm
```

`/xyz` 和 `/pose` 会返回 `object_top_size_mm` 与 `object_top_size_source`，可以用来确认当前坐标用了哪一种烟盒尺寸。`object_top_size_source` 里的 `long_side_mm/short_side_mm` 是类别的真实长短边；`object_top_size_mm` 是本次 PnP 选中朝向后的图像宽高对应尺寸，所以顺序可能变成 `[89.0, 280.0]`。

YOLO 模式会输出：

```text
point_adjustments.left_yolo.confidence   YOLO 置信度
point_adjustments.left_yolo.box_xyxy     YOLO 检测框
point_adjustments.left_yolo.device       实际推理设备，GPU 环境下应为 cuda:0
point_adjustments.left_yolo.points_px    由 YOLO mask 拟合出的四个上表面点
point_adjustments.left_yolo_candidates   左目 YOLO 返回的所有 mask 候选
point_adjustments.yolo_selection         当前用于 PnP 的候选选择规则
debug_images.left_points                 左图最终四点图
```

如果单目画面里有多个烟盒，YOLO 会返回多个 mask。当前接口已经解耦：

```text
left_yolo_candidates = 左目 YOLO 的所有候选 mask
left_yolo            = 被选中用于 PnP/XYZ 的单个候选
center_xyz_mm        = 只根据 left_yolo 这个候选计算
```

查看左目所有候选：

```bash
curl -s http://127.0.0.1:18081/pose | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d['point_adjustments']['left_yolo_candidates'], ensure_ascii=False, indent=2))"
```

查看当前被选中用于算坐标的候选：

```bash
curl -s http://127.0.0.1:18081/pose | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d['point_adjustments']['left_yolo'], ensure_ascii=False, indent=2))"
```

候选里主要看：

```text
candidate_index      当前排序后的候选序号
raw_yolo_index       YOLO 原始返回序号
confidence           YOLO 置信度
score                当前默认排序分数
box_xyxy             检测框
box_center_px        检测框中心
mask_area_px         mask 面积
points_px            由这个 mask 拟合出的四个上表面点
```

默认选择规则是：

```text
--yolo-select score
--yolo-index 0
```

也就是按 `score` 排序后选第 0 个候选算 PnP/XYZ。可以临时指定其它候选，例如选 score 排序后的第 2 个候选：

```bash
curl -s "http://127.0.0.1:18081/xyz?yolo_select=score&yolo_index=1"
```

也可以按位置选择，例如选最左边的候选：

```bash
curl -s "http://127.0.0.1:18081/xyz?yolo_select=leftmost&yolo_index=0"
```

可用选择规则：

```text
score       默认，主要按 YOLO confidence，mask 面积只做很小的辅助加分
largest     mask 面积最大
leftmost    图像里最左
rightmost   图像里最右
topmost     图像里最上
bottommost  图像里最下
center      最靠近图像中心
index       YOLO 原始返回顺序
```

右图目前只作为一致性诊断，最终坐标仍以左图 PnP 为主。如果右图 YOLO 检测失败，脚本会保留左图结果并在 `warnings` 里提示。

## GPU 环境运行

Jetson Orin NX 上已经准备了一个独立的 Python 3.8 GPU 环境：

```bash
/home/unitree/venvs/tv_gpu
```

它安装的是 JetPack 5.1.x / CUDA 11.4 可用的 PyTorch wheel。GPU 模式推荐直接运行：

```bash
cd ~/unifolm-world-model-action/robot_client_unitree_g1_full_20260509/repos/unitree_deploy

bash scripts/cigarette_pose_yolo_gpu.sh
```

等价的完整命令是：

```bash
cd ~/YOLO_YAN_COORDINATION
source /home/unitree/venvs/tv_gpu/bin/activate
PYTHONPATH="$PWD:$PWD/scripts" python scripts/cigarette_pose_optical_api.py \
  --capture \
  --pretty \
  --yolo-device cuda:0
```

如果 `point_adjustments.left_yolo.device` 显示 `cuda:0`，说明当前使用 GPU；如果显示 `cpu`，说明当前 Python 环境没有可用 CUDA PyTorch。

## 常驻 YOLO 服务

如果机器人本体会频繁取坐标，推荐启动常驻服务。它会在启动时加载并预热 YOLO，后续请求复用同一个 Python 进程和同一个 GPU 模型，不会每次重新加载。
服务内部也会复用 `ImageClient`，避免同一进程里反复打开/关闭相机客户端导致 `SubscriberManager is closed`。

前台启动：

```bash
cd ~/YOLO_YAN_COORDINATION

bash scripts/cigarette_pose_yolo_server_gpu.sh
```

如果当前机器没有 CUDA PyTorch，可以临时用 CPU 启动：

```bash
cd ~/YOLO_YAN_COORDINATION
YOLO_DEVICE=cpu bash scripts/cigarette_pose_yolo_server_gpu.sh
```

后台启动：

```bash
cd ~/YOLO_YAN_COORDINATION

nohup bash scripts/cigarette_pose_yolo_server_gpu.sh \
  > /tmp/cigarette_pose_yolo_server.log 2>&1 &

echo $! > /tmp/cigarette_pose_yolo_server.pid
```

默认监听所有网卡：

```text
http://0.0.0.0:18081
```

这里使用 `18081`，因为机器人上 `18080` 已被其它服务占用。

本地电脑和机器人在同一网络时，可以直接打开：

```text
http://192.168.0.149:18081/debug
```

开机自启动：

```bash
cd ~/YOLO_YAN_COORDINATION
sudo cp systemd/cigarette-pose-yolo.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now cigarette-pose-yolo.service
```

查看状态：

```bash
systemctl status cigarette-pose-yolo.service --no-pager
```

查看日志：

```bash
journalctl -u cigarette-pose-yolo.service -f
```

健康检查：

```bash
curl -s http://127.0.0.1:18081/health
```

获取完整 JSON：

```bash
curl -s http://127.0.0.1:18081/pose
```

只获取下游最常用的坐标和距离：

```bash
curl -s http://127.0.0.1:18081/xyz
```

默认 `/xyz` 等价于：

```bash
curl -s "http://127.0.0.1:18081/xyz?yolo_select=score&yolo_index=0"
```

也就是从左目 YOLO 候选里按 `score` 排序，选择第 0 个候选的上表面中心点来算 PnP/XYZ。

只打印 `[x, y, z]`：

```bash
curl -s http://127.0.0.1:18081/xyz | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['center_xyz_mm'])"
```

只打印中心点垂直地面上方 10cm 的点：

```bash
curl -s http://127.0.0.1:18081/xyz | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['center_above_xyz_mm'])"
```

如果临时要改中心点 above 高度，例如 80mm：

```bash
curl -s "http://127.0.0.1:18081/xyz?center_above_height_mm=80" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['center_above_xyz_mm'])"
```

直接看当前输出图片：

一页看全：实时拍照、显示左/右原图、左/右点图、左/右全部候选叠加图、选中的候选、所有 YOLO candidates 和完整 JSON：

```text
http://127.0.0.1:18081/debug
```

本地电脑和机器人在同一网络时，可以直接打开：

```text
http://192.168.0.149:18081/debug
```

如果网络不方便直接访问，也可以从本地电脑通过端口转发访问 `215`，浏览器打开：

```powershell
ssh -L 18082:127.0.0.1:18081 unitree@192.168.0.215
```

这个 SSH 窗口保持打开，然后本地浏览器访问：

```text
http://127.0.0.1:18082/debug
```

如果看 `149`，可以换一个本地端口：

```powershell
ssh -L 18083:127.0.0.1:18081 unitree@192.168.0.149
```

然后打开：

```text
http://127.0.0.1:18083/debug
```

单独看左目结果图：

```text
http://127.0.0.1:18081/debug/left_points.jpg
```

这个端点会实时拍照、YOLO、PnP，并直接返回左目最终四点图。也可以保存到文件：

```bash
curl -s http://127.0.0.1:18081/debug/left_points.jpg -o /tmp/left_points.jpg
```

如果要看三个候选同时画在一张图上的结果：

```text
http://127.0.0.1:18081/debug/left_candidates.jpg
```

通过本地端口转发看 `215` 时：

```text
http://127.0.0.1:18082/debug/left_candidates.jpg
```

候选叠加图会给每个目标画框和四点，并标注：

```text
* #0 XiongMao conf=0.954
```

其中 `*` 表示当前被选中用于 PnP/XYZ 的候选，`#0` 是候选序号，`conf` 是 YOLO 置信度。

其它图片端点：

```text
/debug/left_points.jpg    实时左目结果图
/debug/right_points.jpg   实时右目结果图
/debug/left_input.jpg     实时左目原图
/debug/right_input.jpg    实时右目原图
/debug/left_candidates.jpg  实时左目全部 YOLO candidates 叠加图
/debug/right_candidates.jpg 实时右目全部 YOLO candidates 叠加图
/latest/left_points.jpg   最近一次请求的左目结果图，不重新拍照
/latest/right_points.jpg  最近一次请求的右目结果图，不重新拍照
/latest/left_candidates.jpg  最近一次请求的左目全部候选叠加图，不重新拍照
/latest/right_candidates.jpg 最近一次请求的右目全部候选叠加图，不重新拍照
```

如果有多个烟盒，也可以带选择参数直接看某个候选的点图，例如按最左边的候选画图：

```text
http://127.0.0.1:18081/debug/left_points.jpg?yolo_select=leftmost&yolo_index=0
```

如果临时要按已知直线距离缩放坐标，可以在请求里带参数，例如 750mm：

```bash
curl -s "http://127.0.0.1:18081/xyz?known_range_mm=750"
```

常驻服务返回里会带：

```text
server.resident = true
server.pid
server.request_id
server.elapsed_ms
server.model_cache_size
```

如果 `device` 或 `point_adjustments.left_yolo.device` 是 `cuda:0`，说明当前请求走 GPU。

停止后台服务：

```bash
kill $(cat /tmp/cigarette_pose_yolo_server.pid)
```

## 双目基线诊断

左目和右目镜头表面最凸处距离当前按 `60mm` 记录：

```text
--stereo-baseline-mm 60
```

脚本会额外输出 `stereo_check`，用近似公式做双目视差诊断：

```text
Z = focal_px * baseline_mm / disparity_px
```

注意：这个公式要求左右图已经近似校正/平行，而且 `baseline_mm` 最好是两个 optical center 的距离。当前 60mm 是镜头表面最凸处距离，所以先作为一致性检查，不直接覆盖最终 `center_xyz_mm`。如果 `stereo_check.warning` 出现大偏差，优先检查左右四点是否对应，以及相机是否需要正式双目标定/校正。

## 旧 Mask 模式

旧的 mask 检测只作为回退调试入口保留，正常不要用：

```bash
PYTHONPATH="$PWD:$PWD/scripts" python3 scripts/cigarette_pose_optical_api.py \
  --capture \
  --mode mask \
  --pretty
```

旧 mask 模式仍支持 `--bottom-edge-mode`，但 YOLO 模式不会自动下拉或改动 YOLO 点；YOLO segmentation 没有给出可用 mask 时会直接报错。

调试图里保存：

```text
debug_images.left_points       左图最终四点，机械臂应该使用这个结果
debug_images.right_points      右图最终四点，仅作一致性检查
```

如果未来 keypoint 模型已经直接给出准确的四个上表面点，用 `--mode points`，脚本不会自动改点；它只做排序、PnP 和坐标换算。

如果要离线对比自动修正前后的数值，可以跑：

```bash
PYTHONPATH="$PWD:$PWD/scripts" python3 scripts/cigarette_pose_optical_api.py \
  --capture \
  --bottom-edge-mode none \
  --pretty

PYTHONPATH="$PWD:$PWD/scripts" python3 scripts/cigarette_pose_optical_api.py \
  --capture \
  --bottom-edge-mode auto \
  --pretty
```

`focal-px=260.0` 是当前工作值：2026-05-26 已切到 YOLO segmentation 后，熊猫烟盒按上表面尺寸 `161mm x 95mm` 标定；已知左目到上表面中心点直线距离为 `750mm` 时，连续采样结果为 `759.2 / 747.8 / 748.7mm`，平均约 `751.9mm`，因此默认取 `260.0`。2026-05-27 起利群烟盒按 `280mm x 89mm` 自动切换尺寸。PnP 距离会随物体实际尺寸和 focal 一起缩放；如果之后有新的实测数据，更稳的做法是按不同类别分别测 2-3 个已知距离，再确认统一 focal 是否仍合适。

## 未来 Keypoint 接入

如果之后模型直接输出左图上表面四个角点，可以绕过 segmentation mask，直接用四点算 optical 坐标：

```bash
PYTHONPATH="$PWD:$PWD/scripts" python3 scripts/cigarette_pose_optical_api.py \
  --mode points \
  --left-points '[[341,312],[373,315],[381,364],[344,364]]' \
  --pretty
```

如果同时有真实直线距离，并且想用它来验算或强制缩放结果，可以传入：

```bash
PYTHONPATH="$PWD:$PWD/scripts" python3 scripts/cigarette_pose_optical_api.py \
  --mode points \
  --left-points '[[341,312],[373,315],[381,364],[344,364]]' \
  --known-range-mm 700 \
  --pretty
```

如果同时有右图四点，可以一起传入，用来做左右深度一致性检查：

```bash
PYTHONPATH="$PWD:$PWD/scripts" python3 scripts/cigarette_pose_optical_api.py \
  --mode points \
  --left-points '[[341,312],[373,315],[381,364],[344,364]]' \
  --right-points '[[302,315],[335,315],[340,366],[302,366]]' \
  --pretty
```

也可以传 JSON 文件：

```bash
PYTHONPATH="$PWD:$PWD/scripts" python3 scripts/cigarette_pose_optical_api.py \
  --mode points \
  --left-points /tmp/yolo_left_points.json \
  --pretty
```

支持的 JSON 格式：

```json
[[341, 312], [373, 315], [381, 364], [344, 364]]
```

或者：

```json
{
  "points_px": [[341, 312], [373, 315], [381, 364], [344, 364]]
}
```

## Python 调用示例

如果别的 Python 代码要直接调用：

```python
from cigarette_pose_optical_api import PoseConfig, estimate_pose_from_left_points

points_px = [[341, 312], [373, 315], [381, 364], [344, 364]]
result = estimate_pose_from_left_points(points_px, PoseConfig())

print(result["center_xyz_mm"])
```

运行前建议设置：

```bash
export PYTHONPATH="$PWD:$PWD/scripts"
```

## 判断结果是否可信

不要只看 `ok=true` 或重投影误差。自动调用时优先看这些机器可判断字段：

```text
ok
quality.left_reprojection_ok
quality.right_consistency_ok
left_reprojection_error_px
depth_delta_mm
warnings
point_adjustments
```

`left_points.jpg` 和 `left_points_raw.jpg` 用来离线复查算法，不作为每次上线运行时手动调点的步骤。

经验规则：

```text
优先信 center_xyz_mm / range_from_left_camera_mm / left_depth_mm。
right_depth_mm 只做参考。
如果 depth_delta_mm 很大，先看左右点图，不要直接用深度。
如果 point_adjustments.left_auto_bottom_edge.applied=false，说明这一帧没有触发底边下移。
```

## 烟盒头部往内 1/5 点

`/pose` 和 `/xyz` 会额外返回一个沿烟盒长轴计算的点，以及这个点上方 10cm 的点：

```text
box_head_point_xyz_mm
box_head_point_above_xyz_mm
box_head_one_third_xyz_mm
box_head_one_third_above_xyz_mm
```

其中 `box_head_point_*` 是推荐使用的新字段；`box_head_one_third_*` 只是旧字段名兼容，现在同样返回 1/5 位置。

定义：

```text
1. 用 PnP 得到烟盒上表面的 3D 位姿。
2. 找烟盒长轴的两个端点。
3. 距离左目相机光心更远的那一端定义为“头”。
4. 从“头”沿长轴往烟盒内部走 1/5 长度，得到该点坐标。
5. 再从这个 1/5 点沿地面垂直向上抬高 100mm，得到 above 点。
```

完整调试信息在：

```text
box_head_point
box_head_point_above
box_head_one_third
box_head_one_third_above
```

里面包含：

```text
point_xyz_mm          最终点坐标
head_xyz_mm           远处头部端点
tail_xyz_mm           近处尾部端点
head_range_mm         头部端点到左目光心直线距离
tail_range_mm         尾部端点到左目光心直线距离
long_axis             本次长轴在物体局部坐标里的方向
fraction_from_head    从头部往内走的比例，默认 0.2
distance_from_head_mm 从头部往内走的距离，等于长边长度 / 5
```

`box_head_point_above_xyz_mm` 是在 `box_head_point_xyz_mm` 的基础上沿地面垂直向上抬高，默认高度是 `100mm`，默认角度仍然是 `42.4°`。

只打印头部往内 1/5 点：

```bash
curl -s http://127.0.0.1:18081/xyz \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['box_head_point_xyz_mm'])"
```

只打印这个点上方 10cm 的点：

```bash
curl -s http://127.0.0.1:18081/xyz \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['box_head_point_above_xyz_mm'])"
```

如果要临时改 above 高度，比如改成 `80mm`：

```bash
curl -s "http://127.0.0.1:18081/xyz?box_head_above_height_mm=80" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['box_head_point_above_xyz_mm'])"
```

如果要临时切回 1/3：

```bash
curl -s "http://127.0.0.1:18081/xyz?box_head_fraction_from_head=0.333333" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['box_head_point_xyz_mm'])"
```
## 中心点上方 10cm

如果下游抓取需要“香烟盒中心点上方 10cm”的预抓取点，可以直接用独立转换脚本。默认只做垂直地面方向的 10cm 高度偏移；平行地面的 5cm 偏移先作为可选参数保留。

```bash
cd ~/unifolm-world-model-action/robot_client_unitree_g1_full_20260509/repos/unitree_deploy
python3 scripts/cigarette_pose_above_center.py --pretty
```

默认会请求本机常驻服务：

```text
http://127.0.0.1:18081/xyz
```

默认参数等价于：

```text
height-mm        = 100
ground-offset-mm = 0
x-offset-mm      = 0
```

然后返回：

```text
center_xyz_mm              当前 YOLO/PnP 算出的上表面中心点
above_vertical_only_xyz_mm 只做垂直向上 100mm 后的位置
above_xyz_mm               默认等于垂直向上 100mm 后的位置
vertical_offset_xyz_mm     垂直向上的相机坐标偏移
ground_offset_xyz_mm       平行地面的相机坐标偏移，默认是 [0.0, 0.0, 0.0]
offset_xyz_mm              最终总偏移
```

当前坐标关系：

```text
left_camera_optical:
+X = 图像右方
+Y = 图像下方
+Z = 相机前方/深度

相机 +Z 轴与地面垂直向下方向夹角 = 42.4°
```

默认使用真实垂直方向的 Y-Z 平面向量。脚本里也保留了一个“垂直于 X 轴、平行于地面”的 Y-Z 平面向量，后面需要 5cm 水平偏移时再通过 `--ground-offset-mm` 打开。这个地面方向定义为相机 `+Z` 在地面平面上的投影；如果方向反了，把 `--ground-offset-mm 50` 改成 `--ground-offset-mm -50`。

```text
vertical_down_unit_xyz = [0, sin(42.4°), cos(42.4°)]
vertical_up_unit_xyz   = [0, -sin(42.4°), -cos(42.4°)]
ground_forward_unit_xyz = [0, -cos(42.4°), sin(42.4°)]

100mm 上方偏移约为      [0.0, -67.4, -73.8] mm
默认最终偏移约为        [0.0, -67.4, -73.8] mm

如果手动加 50mm 地面平行偏移：
50mm 地面平行偏移约为   [0.0, -36.9, 33.7] mm
叠加后的最终偏移约为    [0.0, -104.4, -40.1] mm
```

也就是说：

```text
above_x = center_x + x_offset_mm
above_y = center_y - 100 * sin(42.4°)
above_z = center_z - 100 * cos(42.4°)
```

如果之后要叠加平行地面的 5cm：

```bash
python3 scripts/cigarette_pose_above_center.py \
  --ground-offset-mm 50 \
  --pretty
```

如果还要沿相机 `+X` 方向额外偏移，比如 20mm：

```bash
python3 scripts/cigarette_pose_above_center.py \
  --x-offset-mm 20 \
  --pretty
```

只打印上方点 `[x, y, z]`：

```bash
python3 scripts/cigarette_pose_above_center.py \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['above_xyz_mm'])"
```

如果已经有一个中心点坐标，不想重新拍照，可以手动传入：

```bash
python3 scripts/cigarette_pose_above_center.py \
  --xyz=-24.6,107.0,655.3 \
  --pretty
```

如果只是想沿用之前的 `z*cos(42.4°)` 投影校验方式，也提供兼容模式：

```bash
python3 scripts/cigarette_pose_above_center.py \
  --method z_only_projection \
  --pretty
```

这个模式的高度部分只改 `Z`，再叠加 `ground-offset-mm` 指定的地面平行偏移；它不是严格的三维垂直上移，正式抓取预点建议使用默认的 `ground_vertical_yz_vector`。
## 按标签选择最高置信度候选

如果画面里有多种烟盒，可以在 `/xyz` 或 `/pose` 里传 `label` / `yolo_label` / `yolo_class_name`。接口会先按 YOLO 类别过滤候选，再按 `confidence` 从高到低选择第 0 个候选计算坐标。

```bash
curl -s "http://127.0.0.1:18081/xyz?label=XiongMao"
curl -s "http://127.0.0.1:18081/xyz?label=Liqun"
curl -s "http://127.0.0.1:18081/xyz?label=Xizi_Liqun"
```

只打印这个标签下最高置信度目标的中心点：

```bash
curl -s "http://127.0.0.1:18081/xyz?label=XiongMao" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['selected_yolo_label'], d['selected_yolo_confidence'], d['center_xyz_mm'])"
```

常用返回字段：

```text
requested_yolo_label       请求的类别标签
selected_yolo_label        实际选中的 YOLO 类别
selected_yolo_confidence   实际选中候选的 YOLO 置信度
center_xyz_mm              该候选上表面中心点坐标
center_above_xyz_mm        该候选中心点上方 10cm 坐标
box_head_point_above_xyz_mm 该候选头部 1/5 点上方 10cm 坐标
```

`label=Liqun` 会匹配模型类别名 `Xizi_Liqun`；也可以直接传完整类别名。没有匹配到对应标签时，接口会返回错误，并列出当前画面中 YOLO 检到的可用标签。
