# YOLO 烟盒坐标服务

这是机器人左目相机的烟盒 YOLO 检测与坐标输出交付版本。

## 当前版本

- 默认模型：`models/Liqun_Xiongmao.pt`
- 支持类别：`XiongMao`、`Xizi_Liqun`
- 默认设备：Jetson GPU `cuda:0`
- 默认服务端口：`127.0.0.1:18081`
- 坐标系：`left_camera_optical`
  - `+X` 图像右方
  - `+Y` 图像下方
  - `+Z` 相机前方/深度

## 启动服务

在机器人上进入仓库目录：

```bash
cd ~/unifolm-world-model-action/robot_client_unitree_g1_full_20260509/repos/unitree_deploy
nohup bash scripts/cigarette_pose_yolo_server_gpu.sh \
  > /tmp/cigarette_pose_yolo_server.log 2>&1 &
echo $! > /tmp/cigarette_pose_yolo_server.pid
```

检查服务：

```bash
curl -s http://127.0.0.1:18081/health
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

## Debug 页面

机器人本机：

```text
http://127.0.0.1:18081/debug
```

本地电脑 SSH 转发看 `149`：

```powershell
ssh -L 18082:127.0.0.1:18081 unitree@192.168.0.149
```

然后浏览器打开：

```text
http://127.0.0.1:18082/debug
```

更多说明见：

- `docs/cigarette_pose_optical_api.md`
- `docs/yolo_topface_detector_module.md`
