# 门户、吸盘代理、本体控制服务部署

三个独立服务统一在 `/home/unitree/YOLO_YAN_COORDINATION` 运行；服务名、端口和 ROS2 Topic 保持兼容。

| 服务 | 代码 | 启动脚本 | 配置 |
| --- | --- | --- | --- |
| `g1d-tool-portal.service`，18079 | `services/g1d_tool_portal/` | `scripts/g1d_tool_portal_server.sh` | `config/g1d_tool_portal.env` |
| `suction-api-proxy.service`，18080 | `services/es80z_api_proxy/` | `scripts/suction_api_proxy_server.sh` | `config/suction_api_proxy.env` |
| `g1d-body-control.service`，ROS2 | `services/g1d_body_control/` | `scripts/g1d_body_control.sh` | `config/g1d_body_control.env` |

## 依赖

门户和吸盘代理仅依赖系统 Python 3 标准库，不需要 Conda、dotenv 或 Node.js。页面资源在各自 `static/` 目录中，修改后重启服务。

本体控制依赖系统 Python 对应的 ROS2 Foxy（`rclpy`、`std_msgs`）、CycloneDDS RMW，以及已编译的 `g1d_simple_control`、`g1d_height_control`。立柱状态接口默认是本机 28089。安装脚本不会安装这些依赖。本体控制保留原来的真实控制行为，无新增预览模式；只在准备好接收 ROS2 控制任务时启动。

## 首次配置

在机器人上执行以下命令。已有配置时不要重新复制模板覆盖它。

```bash
cd /home/unitree/YOLO_YAN_COORDINATION
cp -n config/g1d_tool_portal.env.example config/g1d_tool_portal.env
cp -n config/suction_api_proxy.env.example config/suction_api_proxy.env
cp -n config/g1d_body_control.env.example config/g1d_body_control.env
```

逐个编辑 `.env`：

- 门户：`G1D_BOX_PUBLIC_HOST` 是浏览器访问盒子的 IP；`G1D_BOX_INTERNAL_HOST` 是本体探测盒子服务使用的 IP。
- 吸盘代理：`SUCTION_PROXY_UPSTREAM` 是盒子 ES80Z API，例如 `http://192.168.123.5:18089`。与门户内网地址保持一致。
- 本体控制：核对 `G1D_BODY_INTERFACE`、两个 SDK 程序路径、立柱接口、ROS 环境路径和 Topic。需要时明确配置 `CYCLONEDDS_URI`。

`.env.example` 提交到 Git；实际 `.env` 已加入 `.gitignore`，每台机器人单独配置，更新代码不会覆盖。文件请使用 UTF-8、LF 换行、简单的 `KEY=value` 格式；不写 `export`、不引用其他变量、不使用命令替换。配置会被启动脚本 source，应只由可信的部署人员编辑。

## 安装 / 从旧目录迁移

```bash
cd /home/unitree/YOLO_YAN_COORDINATION
bash scripts/install_autostart_services.sh portal suction-proxy body-control
```

支持单独指定 `portal`、`suction-proxy`、`body-control`。不传参数保持原来五个服务的安装行为；`all` 安装全部八个服务。

安装器先检查配置，缺少时仅创建模板并退出，不改变 systemd 服务。填写配置后重新运行。已有 `.env` 保持不覆盖。标准部署路径和账号是 `/home/unitree/YOLO_YAN_COORDINATION`、`unitree`；其他路径或账号需手动调整 service 模板并安装。

安装器将现有 service 和存在的附加配置备份到 `/var/backups/yolo-services/<时间>-<进程号>/`，保留现有附加配置，然后安装新模板、执行 daemon-reload、enable 和 restart。三个集成服务即使已运行也会重启，以切换源码路径；本体控制重启应安排在无动作任务时。部署前用 `systemctl cat <服务名>` 核对附加配置，旧的 ExecStart 或 EnvironmentFile 覆盖项需人工调整。

旧目录 `/opt/g1d-tool-portal`、`/home/unitree/es80z_api_proxy`、`/home/unitree/g1d_body_control` 不会被删除。服务切换后使用仓库源码，不需要再复制 Python 文件到旧目录。

## 配置加载与手动运行

systemd 的 `EnvironmentFile` 读取仓库内 `.env`，变量通过进程环境传给 Python。启动脚本也加载同一文件，保证手动启动行为一致。配置文件是必需项，缺失时拒绝启动。Python 参数优先级为：命令行参数 > 环境变量 > 原有默认值。

手动运行前确认对应端口或 ROS 节点没有另一个服务占用：

```bash
bash scripts/g1d_tool_portal_server.sh
bash scripts/suction_api_proxy_server.sh
# 此项是真实控制 ROS2 节点，不是预览：
bash scripts/g1d_body_control.sh
```

仅修改 `.env` 时重启对应服务即可；修改 `.service` 时先执行 `sudo systemctl daemon-reload`。本体控制使用 `/usr/bin/python3`，避免激活 YOLO Conda 环境后找不到 ROS 扩展。

## 验证

```bash
systemctl status g1d-tool-portal suction-api-proxy g1d-body-control --no-pager
curl http://127.0.0.1:18079/health
curl http://127.0.0.1:18079/api/catalog
curl http://127.0.0.1:18080/api/v1/health
journalctl -u g1d-body-control -n 50 --no-pager
```

门户的工具名、端口和路径在 `services/g1d_tool_portal/catalog.py` 中。门户只做健康检查和链接跳转。吸盘代理保留原 GET/POST、路径、请求体和上游错误状态码；它会真实转发控制请求。

ROS2 节点名保持 `g1d_body_control_api`，默认命令 Topic 为 `/g1d_body_control/task_command`、状态 Topic 为 `/g1d_body_control/task_status`，消息类型为 `std_msgs/msg/String` JSON。可在与服务相同的 ROS/DDS 环境下执行 `ros2 node list` 检查发现，不必发送动作命令。

仓库测试使用本地模拟 HTTP 上游，不访问机器人：

```bash
python -m unittest discover -s tests -p 'test_integrated*.py'
```

## 回滚

将备份目录中的对应 `.service` 恢复到 `/etc/systemd/system/`，核对附加配置，然后执行 daemon-reload 和 restart。旧代码目录保持存在，可恢复原部署路径。本机 `.env` 不参与旧版本运行，保留供后续迁移使用。
