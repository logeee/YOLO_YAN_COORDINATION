# G1D 位置微调服务

这个服务只负责调用底盘 SDK 做微调，和 YOLO 坐标服务分开运行。

- YOLO 坐标服务：`18081`
- 位置微调服务：`18084`
- 默认绑定：`0.0.0.0`
- 默认目标：靠近机器人那条边的地面前向距离为 `200mm`
- 默认 SDK：`/home/unitree/unitree_sdk2/build/bin/g1d_simple_control`

## 开机启动

```bash
cd ~/YOLO_YAN_COORDINATION
sudo cp systemd/g1d-pose-adjust.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now g1d-pose-adjust.service
```

检查状态：

```bash
systemctl status g1d-pose-adjust.service --no-pager
curl -s http://127.0.0.1:18084/health
```

查看日志：

```bash
journalctl -u g1d-pose-adjust.service -f
```

## 推荐调用

先做预检，不会动机器人：

```bash
curl -s "http://127.0.0.1:18084/adjust?dry_run=1"
```

确认现场安全后，一步微调：

```bash
curl -s http://127.0.0.1:18084/adjust
```

从同一局域网电脑访问 149 机器人：

```bash
curl -s "http://192.168.60.121:18084/adjust?dry_run=1"
curl -s http://192.168.60.121:18084/adjust
```

`/adjust` 的内部流程：

```text
1. 读取 YOLO /xyz。
2. 根据 robot_alignment.control_hint.box_parallel_yaw_deg 先 turn_left / turn_right。
3. 转向后等待短暂稳定时间。
4. 重新读取 YOLO /xyz。
5. 根据 near_edge_robot_alignment.target.ground_forward_mm 做 forward / back。
6. 再读取一次 YOLO /xyz，返回 final_plan。
```

如果第一段 SDK 命令失败，服务会直接返回错误，不继续执行下一段。

## 调试接口

只看计划，不会动机器人：

```bash
curl -s http://127.0.0.1:18084/plan
```

只看下一步动作，不会动机器人：

```bash
curl -s http://127.0.0.1:18084/step
```

确认执行一个小动作：

```bash
curl -s "http://127.0.0.1:18084/step?confirm=1"
```

紧急停止：

```bash
curl -s "http://127.0.0.1:18084/stop?confirm=1"
```

## 常用参数

临时指定目标距离，例如近端边前向距离改成 `220mm`：

```bash
curl -s "http://127.0.0.1:18084/adjust?target_near_edge_forward_mm=220"
```

指定 YOLO 标签：

```bash
curl -s "http://127.0.0.1:18084/adjust?label=XiongMao"
curl -s "http://127.0.0.1:18084/adjust?label=Xizi_Liqun"
```

调小速度：

```bash
curl -s "http://127.0.0.1:18084/adjust?turn_speed=0.08&drive_speed=0.08"
```

修改容差：

```bash
curl -s "http://127.0.0.1:18084/adjust?yaw_tolerance_deg=1.5&distance_tolerance_mm=10"
```

## 返回值怎么看

`stages` 里有两个阶段：

```text
turn          转向阶段
forward_back  前后移动阶段
```

每个阶段里重点看：

```text
metrics.box_parallel_yaw_deg       烟盒长轴相对机器人需要修正的角度
metrics.near_edge_forward_mm       靠近机器人那条边的当前前向距离
metrics.distance_error_mm          当前前向距离 - 目标距离
command.action                     本阶段要执行的动作
execution.ok                       SDK 命令是否成功
```

`final_plan` 是动作执行完后重新拍照算出的结果。若 `final_plan.command.action` 是 `none`，说明角度和距离都已经在容差内。
