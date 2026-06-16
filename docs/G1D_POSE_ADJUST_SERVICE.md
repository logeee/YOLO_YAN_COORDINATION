# G1D 位置微调服务

微调服务只负责调用底盘 SDK。YOLO 坐标服务是 `18081`，微调服务是 `18084`。

## 开机启动

```bash
cd ~/YOLO_YAN_COORDINATION
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

## 常用调用

预检，不会动：

```bash
curl -s "http://127.0.0.1:18084/adjust?dry_run=1"
curl -s "http://192.168.60.121:18084/adjust?dry_run=1"
```

执行一次微调：

```bash
curl -s http://127.0.0.1:18084/adjust
curl -s http://192.168.60.121:18084/adjust
```

实时调试输出：

```bash
curl -N -s "http://127.0.0.1:18084/adjust?stream=1"
curl -N -s "http://192.168.60.121:18084/adjust?stream=1"
```

`stream=1` 会返回 NDJSON，一行一个事件：

```text
adjust_started
plan_ready
command_started
command_finished
adjust_finished
```

急停：

```bash
curl -s "http://127.0.0.1:18084/stop?confirm=1"
```

## 可选参数

```bash
# 指定目标距离，默认 200mm
curl -s "http://127.0.0.1:18084/adjust?target_near_edge_forward_mm=220"

# 指定烟盒类别
curl -s "http://127.0.0.1:18084/adjust?label=XiongMao"
curl -s "http://127.0.0.1:18084/adjust?label=Xizi_Liqun"
curl -s "http://127.0.0.1:18084/plan?label=XiongMao"

# 调速度
curl -s "http://127.0.0.1:18084/adjust?turn_speed=0.08&drive_speed=0.08"
```

`label` 会原样传给 YOLO `/xyz`，YOLO 会先按类别选目标，再把该目标的四点、角度、近端边距离交给微调服务。`label=Liqun` 也可以匹配 `Xizi_Liqun`。

## 返回值

重点看这些字段：

```text
requested_yolo_label                    这次请求指定的 YOLO 标签
selected_yolo_label                     YOLO 实际选中用于微调的标签
yolo_label_matched                      指定标签和选中标签是否匹配
metrics.box_parallel_yaw_deg           当前烟盒长轴角
metrics.near_edge_forward_mm           当前近端边前向距离
metrics.control_distance_error_mm      实际用于 forward/back 的距离误差
commands                               本次会执行的 SDK 命令
executions[].ok                        每条 SDK 命令是否成功
```

说明：

- `/adjust` 只读取一次 YOLO `/xyz`，然后发一批控制命令。
- 动作顺序是先 `turn_left/turn_right`，再 `forward/back`。
- 前进距离会先按计划转向角预测转向后的近端边距离，再计算到目标距离还差多少。
- 不做闭环复测；需要复测时再次调用 `/adjust`。
