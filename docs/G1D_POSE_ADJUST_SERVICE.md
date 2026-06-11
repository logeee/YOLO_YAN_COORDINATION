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
1. 读取一次 YOLO /xyz。
2. 根据这一次 robot_alignment.control_hint.box_parallel_yaw_deg 计算 turn_left / turn_right。
3. 根据这一次 near_edge_robot_alignment.target.ground_forward_mm 计算 forward / back。
4. 按这一次计算结果依次发控制命令。
5. 中途不会重新拍照，也不会做闭环修正。
```

如果某个 SDK 命令失败，服务会直接返回错误，不继续执行后续命令。

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

控制时长按物理量计算：

```text
转向时长 = abs(box_parallel_yaw_deg) 转成 rad 后 / turn_speed
前后时长 = abs(distance_error_mm) / 1000 / drive_speed
默认最长单条命令 5s，可用 max_duration_sec 临时修改
```

因为 `/adjust` 是先转向再前进，但只拍一次照，所以前进距离要按“转向后的坐标系”计算。公式是确定性的，不使用经验系数：

```text
predicted_after_turn_forward_mm = near_edge_forward_mm * cos(planned_turn_yaw_rad)
                                  - near_edge_right_mm * sin(planned_turn_yaw_rad)

control_distance_error_mm = predicted_after_turn_forward_mm - target_near_edge_forward_mm
```

返回 JSON 里：

```text
distance_error_mm              原始近端边距离误差
control_distance_error_mm      实际用于 forward/back 的距离误差
planned_turn_yaw_deg           本次计划转向角
predicted_after_turn_forward_mm 计划转向后的近端边前向距离
forward_delta_from_planned_turn_mm 转向导致的前向距离变化量
```

修改容差：

```bash
curl -s "http://127.0.0.1:18084/adjust?yaw_tolerance_deg=1.5&distance_tolerance_mm=10"
```

## 返回值怎么看

`stages` 里只有一次计算和一批控制：

```text
single_calculation_control
```

重点看：

```text
metrics.box_parallel_yaw_deg       烟盒长轴相对机器人需要修正的角度
metrics.near_edge_forward_mm       靠近机器人那条边的当前前向距离
metrics.distance_error_mm          当前前向距离 - 目标距离
commands                            这一次计算得到的控制命令列表
executions.ok                       每个 SDK 命令是否成功
```

`final_plan` 固定为 `null`，因为 `/adjust` 不再执行后置重拍和闭环判断。
