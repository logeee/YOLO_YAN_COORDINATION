# G1-D 蓝牙遥控

这个服务用于无 WiFi 场景下代替遥控手柄。机器人作为 BLE 外设广播 `G1D-BLE-RCS-<编号>`，手机小程序作为 BLE 中心设备连接后写入短命令。

## 设备检查结论

当前 149 机器人已经确认具备 BLE 服务条件：

```text
Ubuntu 20.04
BlueZ 5.53
hci0 Realtek USB Bluetooth
bluetooth.service active
LEAdvertisingManager1 available
GattManager1 available
python3 dbus OK
python3 gi OK
```

## BLE 协议

服务 UUID：

```text
6f9d0001-7f70-4f8f-9f25-41f0a7a1b001
```

写入控制 Characteristic：

```text
6f9d0002-7f70-4f8f-9f25-41f0a7a1b001
```

读取状态 Characteristic：

```text
6f9d0003-7f70-4f8f-9f25-41f0a7a1b001
```

推荐每台机器人配置独立广播名，例如：

```text
G1D-BLE-RCS-12700
G1D-BLE-RCS-1001
G1D-BLE-RCS-123456
```

小程序只会扫描并列出 `G1D-BLE-RCS-` 后 4 到 6 位数字的设备，点选后再连接。小程序长按时每 250ms 写一次：

```text
H forward 0.20
H back 0.20
H turn_left 0.20
H turn_right 0.20
H column_up 0.05
H column_down 0.05
```

松手、断开前、急停时写：

```text
S
```

机器人端有看门狗，默认超过 `0.6s` 没收到 hold 心跳就自动执行 stop。

## 本地检查

Windows 本地没有 BlueZ，不能真正启动 BLE 广播，但可以做语法检查：

```powershell
python -m py_compile scripts\g1d_ble_remote_server.py
```

## 机器人预览模式

先不要动机器人，只验证手机能扫描、连接、写入命令：

```bash
cd /home/unitree/YOLO_YAN_COORDINATION
bash scripts/g1d_ble_remote_server.sh
```

查看日志：

```bash
tail -f /tmp/g1d_ble_remote_service.log
```

或者前台启动时直接看终端输出。预览模式收到 `H forward 0.20` 只会打印：

```text
preview hold forward 0.20
```

## 小程序示例

示例目录：

```text
miniprogram/g1d_ble_remote
```

用微信开发者工具导入这个目录。页面会扫描名称为 `G1D-BLE-RCS-<编号>` 的 BLE 设备，连接后查找上面的服务 UUID 和控制 Characteristic。

按钮行为：

```text
长按前进/后退/左转/右转/升/降：每 250ms 写一次 H <action> <speed>
松手：写 S
STOP：写 S
```

小程序调试时先配合机器人预览模式，看机器人日志是否能收到对应动作。

## 机器人执行模式

确认 BLE 连接和写入稳定后，再启用真实控制：

```bash
cd /home/unitree/YOLO_YAN_COORDINATION
G1D_BLE_EXECUTE_ENABLED=1 bash scripts/g1d_ble_remote_server.sh
```

执行模式会调用：

```text
/home/unitree/unitree_sdk2/build/bin/g1d_simple_control eth0 forward <speed> 600
/home/unitree/unitree_sdk2/build/bin/g1d_simple_control eth0 back <speed> 600
/home/unitree/unitree_sdk2/build/bin/g1d_simple_control eth0 turn_left <speed> 600
/home/unitree/unitree_sdk2/build/bin/g1d_simple_control eth0 turn_right <speed> 600
/home/unitree/unitree_sdk2/build/bin/g1d_simple_control eth0 up <speed> 600
/home/unitree/unitree_sdk2/build/bin/g1d_simple_control eth0 down <speed> 600
/home/unitree/unitree_sdk2/build/bin/g1d_simple_control eth0 stop
```

## 开机自启

```bash
cd /home/unitree/YOLO_YAN_COORDINATION
sudo cp systemd/g1d-ble-remote.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now g1d-ble-remote.service
```

检查：

```bash
systemctl is-enabled g1d-ble-remote.service
systemctl is-active g1d-ble-remote.service
journalctl -u g1d-ble-remote.service -f
```

停止：

```bash
sudo systemctl stop g1d-ble-remote.service
```

## 安全点

- 先用预览模式验证手机写入命令。
- 执行模式第一次测试速度保持 `0.05` 到 `0.10`。
- 小程序长按必须持续发心跳，松手必须写 `S`。
- 机器人端断连或收不到心跳会自动 stop。
- 新动作开始前会清理遗留的 `g1d_simple_control eth0` 控制进程。
