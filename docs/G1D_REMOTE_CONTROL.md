# G1-D 遥控页面

这个页面用于在遥控器损坏时做临时手动控制。代码放在本仓库，默认预览模式不会调用机器人 SDK，适合先在本地电脑测试页面和按钮逻辑。

## 本地测试

在本地仓库启动：

```bash
python scripts/g1d_remote_control_server.py --bind 127.0.0.1 --port 18086
```

打开：

```text
http://127.0.0.1:18086/
```

页面显示“预览模式”时，点击前进、后退、左转、右转、升、降只会返回将要执行的命令，不会动机器人。

## 部署到机器人

不要放到官方 SDK 目录 `/home/unitree/unitree_sdk2`。建议仍然放在：

```text
/home/unitree/YOLO_YAN_COORDINATION
```

预览模式：

```bash
cd /home/unitree/YOLO_YAN_COORDINATION
bash scripts/g1d_remote_control_server.sh
```

执行模式：

```bash
cd /home/unitree/YOLO_YAN_COORDINATION
REMOTE_CONTROL_EXECUTE_ENABLED=1 bash scripts/g1d_remote_control_server.sh
```

打开：

```text
http://<机器人当前IP>:18086/
```

## SDK 命令

页面默认调用同一个控制程序：

```text
/home/unitree/unitree_sdk2/build/bin/g1d_simple_control eth0 stop
/home/unitree/unitree_sdk2/build/bin/g1d_simple_control eth0 up <speed> <duration>
/home/unitree/unitree_sdk2/build/bin/g1d_simple_control eth0 down <speed> <duration>
/home/unitree/unitree_sdk2/build/bin/g1d_simple_control eth0 forward <speed> <duration>
/home/unitree/unitree_sdk2/build/bin/g1d_simple_control eth0 back <speed> <duration>
/home/unitree/unitree_sdk2/build/bin/g1d_simple_control eth0 turn_left <speed> <duration>
/home/unitree/unitree_sdk2/build/bin/g1d_simple_control eth0 turn_right <speed> <duration>
```

按住按钮时会启动一条默认 600 秒的长时运动命令；松手、移出按钮或按 `STOP` 都会发送停止命令，用于截断当前运动。

执行模式下，每次启动新动作或 STOP 前，服务都会先清理可能遗留的 `g1d_simple_control <interface>` 进程，再执行新命令或 `stop`。这样即使服务重启后遗留了旧的 600 秒控制进程，也会在下一次操作时尽量清掉。

## 安全建议

- 先用预览模式确认每个按钮返回的 `argv`。
- 执行模式下速度先保持默认低速。
- 操作时人远离机器人和升降柱运动范围。
- 页面 `STOP` 会调用 `g1d_simple_control eth0 stop`。
