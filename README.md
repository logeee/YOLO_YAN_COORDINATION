# YOLO 烟盒坐标服务

这是机器人左目相机的烟盒 YOLO 检测与坐标输出交付版本。

## 当前版本

- 默认模型：`models/Liqun_Xiongmao.pt`
- 支持类别：`XiongMao`、`Xizi_Liqun`
- 默认设备：Jetson GPU `cuda:0`
- 默认服务地址：`0.0.0.0:18081`
- 坐标系：`left_camera_optical`
  - `+X` 图像右方
  - `+Y` 图像下方
  - `+Z` 相机前方/深度

## 部署到机器人

在机器人上拉取仓库：

```bash
cd ~
git clone https://github.com/logeee/YOLO_YAN_COORDINATION.git
cd ~/YOLO_YAN_COORDINATION
```

如果已经装过，更新到最新版：

```bash
cd ~/YOLO_YAN_COORDINATION
git pull
```

服务文件默认要求仓库路径是：

```text
/home/unitree/YOLO_YAN_COORDINATION
```

## 临时启动服务

在机器人上进入仓库目录：

```bash
cd ~/YOLO_YAN_COORDINATION
nohup bash scripts/cigarette_pose_yolo_server_gpu.sh \
  > /tmp/cigarette_pose_yolo_server.log 2>&1 &
echo $! > /tmp/cigarette_pose_yolo_server.pid
```

临时停止：

```bash
kill $(cat /tmp/cigarette_pose_yolo_server.pid)
```

检查服务是否启动：

```bash
curl -s http://127.0.0.1:18081/health
```

如果当前机器没有 CUDA PyTorch，可以临时用 CPU 启动：

```bash
YOLO_DEVICE=cpu bash scripts/cigarette_pose_yolo_server_gpu.sh
```

本地电脑和机器人在同一网络时，可以直接访问：

```text
http://<机器人IP>:18081/debug
```

例如当前 G1D 本体无线 IP：

```text
http://192.168.60.121:18081/debug
```

## 开机启动

推荐正式部署时使用 systemd。这样机器人断电重启后，YOLO 坐标服务会自动恢复。

安装并立即启动：

```bash
cd ~/YOLO_YAN_COORDINATION
sudo cp systemd/cigarette-pose-yolo.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now cigarette-pose-yolo.service
```

确认已经设置为开机自启动：

```bash
systemctl is-enabled cigarette-pose-yolo.service
```

正常应该输出：

```text
enabled
```

确认当前正在运行：

```bash
systemctl is-active cigarette-pose-yolo.service
```

正常应该输出：

```text
active
```

查看详细状态：

```bash
systemctl status cigarette-pose-yolo.service --no-pager
```

查看实时日志：

```bash
journalctl -u cigarette-pose-yolo.service -f
```

重启服务：

```bash
sudo systemctl restart cigarette-pose-yolo.service
```

停止服务：

```bash
sudo systemctl stop cigarette-pose-yolo.service
```

取消开机自启动：

```bash
sudo systemctl disable --now cigarette-pose-yolo.service
```

## CPU / GPU 设备选择

默认使用 GPU：

```text
YOLO_DEVICE=cuda:0
```

如果当前机器没有 CUDA PyTorch，可以给 systemd 加 CPU override：

```bash
sudo mkdir -p /etc/systemd/system/cigarette-pose-yolo.service.d
sudo tee /etc/systemd/system/cigarette-pose-yolo.service.d/override.conf >/dev/null <<'EOF'
[Service]
Environment=YOLO_DEVICE=cpu
EOF
sudo systemctl daemon-reload
sudo systemctl restart cigarette-pose-yolo.service
```

如果后续装好了 CUDA PyTorch，要切回 GPU：

```bash
sudo rm -f /etc/systemd/system/cigarette-pose-yolo.service.d/override.conf
sudo systemctl daemon-reload
sudo systemctl restart cigarette-pose-yolo.service
```

检查当前实际使用的设备：

```bash
curl -s http://127.0.0.1:18081/health
```

看返回里的：

```text
requested_device
resolved_device
cuda_available
```

## 取坐标

完整坐标：

```bash
curl -s http://127.0.0.1:18081/xyz
```

只取抓取预点，当前定义为“远处头部往内走 1/5，再垂直地面向上 10cm”：

```bash
curl -s http://127.0.0.1:18081/xyz \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['box_head_point_above_xyz_mm'])"
```

只取中心点垂直地面向上 10cm 的点：

```bash
curl -s http://127.0.0.1:18081/xyz \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['center_above_xyz_mm'])"
```

相关字段：

```text
center_xyz_mm                 烟盒上表面中心点
center_above_xyz_mm           中心点垂直地面向上 10cm 的点
box_head_point_xyz_mm          远处头部往内 1/5 的点
box_head_point_above_xyz_mm    上面这个点垂直地面向上 10cm 的点
range_from_left_camera_mm      左目光心到中心点直线距离
```

临时切回 1/3：

```bash
curl -s "http://127.0.0.1:18081/xyz?box_head_fraction_from_head=0.333333"
```

临时修改 above 高度，例如 80mm：

```bash
curl -s "http://127.0.0.1:18081/xyz?box_head_above_height_mm=80"
```

临时修改中心点 above 高度，例如 80mm：

```bash
curl -s "http://127.0.0.1:18081/xyz?center_above_height_mm=80"
```

## 149 机器人附加推荐

149 这台机器人的电机目前有实测偏差，下面是临时推荐给抓取端使用的点位规则。单位都是 `mm`，坐标系仍然是 `left_camera_optical`。

```text
XiongMao:
  使用 box_head_point_above_xyz_mm，也就是 head_1_5_above
  然后额外加偏移 [20, 0, 0]

Xizi_Liqun:
  直接使用 center_above_xyz_mm
```

熊猫烟盒推荐点：

```bash
curl -s "http://127.0.0.1:18081/xyz?label=XiongMao" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); p=d['box_head_point_above_xyz_mm']; p=[p[0]+20.0,p[1],p[2]]; print(p)"
```

利群烟盒推荐点：

```bash
curl -s "http://127.0.0.1:18081/xyz?label=Xizi_Liqun" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['center_above_xyz_mm'])"
```

`label=Liqun` 也可以匹配 `Xizi_Liqun`：

```bash
curl -s "http://127.0.0.1:18081/xyz?label=Liqun" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['center_above_xyz_mm'])"
```

## 方向、距离和机器人微调量

`/xyz` 会返回 `robot_alignment`，这是根据 YOLO/PnP 的四点和中心点换算出来的机器人微调参考量。它只做感知计算，不会直接发布运动命令。

直接查看完整结果：

```bash
curl -s http://127.0.0.1:18081/xyz \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d['robot_alignment'], ensure_ascii=False, indent=2))"
```

只打印最常用的角度和距离：

```bash
curl -s http://127.0.0.1:18081/xyz \
  | python3 -c "import sys,json; d=json.load(sys.stdin); a=d['robot_alignment']; t=a['target']; c=a['control_hint']; print('range_mm=', t['range_from_left_camera_mm'], 'forward_m=', c['forward_distance_m'], 'right_m=', c['lateral_error_m'], 'turn_deg=', c['turn_first_yaw_deg'], 'box_yaw_deg=', c['box_parallel_yaw_deg'])"
```

常用字段：

```text
selected_orientation                              当前自动选中的横/竖物理尺寸假设
robot_alignment.target.range_from_left_camera_mm   左目光心到上表面中心点的直线距离
robot_alignment.target.ground_forward_mm           按相机安装角 42.4° 投影后的地面前向距离
robot_alignment.target.right_mm                    目标相对左目/机器人向右的偏差
robot_alignment.target.bearing_right_deg           目标相对正前方偏右多少度
robot_alignment.target.cmd_vel_yaw_to_center_deg   要先朝目标转多少度，正号按 ROS angular.z 左转约定
robot_alignment.box_axis.axis_yaw_mod180_deg       烟盒长轴相对机器人正前方的无向夹角
robot_alignment.control_hint.forward_distance_m    后续 SDK 可用的前进距离参考
robot_alignment.control_hint.lateral_error_m       横向误差参考，当前 G1D SDK 没有直接传 lateral Move
robot_alignment.control_hint.height_down_m         目标相对相机的垂直向下距离参考
```

G1D ROS1 SDK 里默认订阅 `/cmd_vel`。当前 SDK 代码实际把 `Twist.linear.x` 作为前后速度、`Twist.angular.z` 作为转向速度传给底层 `Move(vx, 0, yaw)`，所以这一版先输出角度和距离，不自动动机器人。后面做闭环时建议：

```text
先用 turn_first_yaw_deg / turn_first_yaw_rad 修正朝向
再用 forward_distance_m 做前后距离闭环
height_down_m 用于手臂或机身高度微调
box_parallel_yaw_deg 用于让机器人或末端执行器和烟盒长轴对齐
```

当前会同时计算两种物理尺寸假设：

```text
long_x_short_y   四点排序里的 0-1 / 3-2 方向是烟盒物理长边
short_x_long_y   四点排序里的 1-2 / 0-3 方向是烟盒物理长边
```

`selected_orientation` 是系统根据重投影误差和左右目深度一致性自动选出的假设。两套假设的微调结果都会放在 `robot_alignment_hypotheses`，可以这样查看：

```bash
curl -s http://127.0.0.1:18081/xyz \
  | python3 -c "import sys,json; d=json.load(sys.stdin); hs=d['robot_alignment_hypotheses']; [print(k, 'selected=', v['selected'], 'range=', v['range_from_left_camera_mm'], 'turn=', v['robot_alignment']['control_hint']['turn_first_yaw_deg'], 'box_yaw=', v['robot_alignment']['control_hint']['box_parallel_yaw_deg']) for k,v in hs.items()]"
```

## Debug 页面

本地电脑和机器人在同一网络时，直接打开：

```text
http://<机器人IP>:18081/debug
```

例如当前 G1D 本体无线 IP：

```text
http://192.168.60.121:18081/debug
```

机器人本机也可以打开：

```text
http://127.0.0.1:18081/debug
```

如果网络不方便直连，再用 SSH 转发看 `149`：

```powershell
ssh -L 18082:127.0.0.1:18081 unitree@192.168.0.149
```

然后浏览器打开：

```text
http://127.0.0.1:18082/debug
```

说明：

- `/debug` 每刷新一次都会重新拍照、YOLO 推理并计算坐标。
- 如果当前服务是 CPU 模式，刷新会明显慢一些；看 `/health` 里的 `resolved_device` 可以确认当前是 `cpu` 还是 `cuda:0`。
- 页面图片会绑定本次请求的 `request_id`，避免浏览器缓存或连续刷新时看到旧图。

更多说明见：

- `docs/cigarette_pose_optical_api.md`
- `docs/yolo_topface_detector_module.md`
- `docs/G1D_POSE_ADJUST_SERVICE.md`

## G1D 位置微调服务

微调服务单独跑在 `18084`，YOLO 服务仍然跑在 `18081`。

手动启动或重启：

```bash
cd ~/YOLO_YAN_COORDINATION
bash scripts/g1d_pose_adjust_service.sh
```

开机自启动：

```bash
sudo cp systemd/g1d-pose-adjust.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now g1d-pose-adjust.service
```

检查：

```bash
systemctl is-enabled g1d-pose-adjust.service
systemctl is-active g1d-pose-adjust.service
curl -s http://127.0.0.1:18084/health
```

一键微调预检，不会动机器人：

```bash
curl -s "http://127.0.0.1:18084/adjust?dry_run=1"
curl -s "http://192.168.60.121:18084/adjust?dry_run=1"
```

一键微调，会真的调用底盘 SDK：

```bash
curl -s http://127.0.0.1:18084/adjust
curl -s http://192.168.60.121:18084/adjust
```

实时调试输出，一行一个 JSON 事件：

```bash
curl -N -s "http://127.0.0.1:18084/adjust?stream=1"
curl -N -s "http://192.168.60.121:18084/adjust?stream=1"
```

紧急停止：

```bash
curl -s "http://127.0.0.1:18084/stop?confirm=1"
```

常用参数：

```bash
curl -s "http://127.0.0.1:18084/adjust?target_near_edge_forward_mm=220"
curl -s "http://127.0.0.1:18084/adjust?label=XiongMao"
curl -s "http://127.0.0.1:18084/adjust?turn_speed=0.08&drive_speed=0.08"
```

更多说明见 `docs/G1D_POSE_ADJUST_SERVICE.md`。

## G1-D 烟盒相对位置可视化

这是独立页面，不改 YOLO 服务。启动：

```bash
cd ~/YOLO_YAN_COORDINATION
bash scripts/g1d_cigarette_visualizer_server.sh
```

打开：

```text
http://127.0.0.1:18085/
http://<机器人IP>:18085/
```

当前没有 URDF 引用的 `meshes/*.STL`，页面会显示 G1-D 骨架和代理体。烟盒厚度没有实测值，默认 `20mm`，页面里可以改。

更多说明见 `docs/G1D_CIGARETTE_VISUALIZER.md`。
