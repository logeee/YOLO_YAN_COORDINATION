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

更多说明见：

- `docs/cigarette_pose_optical_api.md`
- `docs/yolo_topface_detector_module.md`
