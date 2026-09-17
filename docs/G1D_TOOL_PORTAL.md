# G1-D 工具门户

源码：`services/g1d_tool_portal/`；默认端口：`18079`；服务名：`g1d-tool-portal.service`。

## 页面内容

门户分为三组：

- 调试前端：吸盘、YOLO、SLAM 动作链、3D 可视化、遥控和机械臂控制台。
- 核心运行服务：机械臂 IK 等不单独占用前端端口、但需要确认真实运行模式的服务。
- 接口服务：位置微调、立柱高度、导航兼容层和机械臂命令转发等 API。

门户不仅检查端口，还会读取各服务已有的健康 JSON，选择性展示版本、模型、运行模式和 feature。机械臂联合 IK 使用盒子 `18090 /api/state` 的实时 ROS2 状态，因此页面显示的是当前节点实际发布的 `waist_ik_mode`、控制关节数、连续轨迹和求解参数，不是静态说明文字。

## 配置

从 `config/g1d_tool_portal.env.example` 创建不受 Git 跟踪的 `config/g1d_tool_portal.env`：

```bash
cp -n config/g1d_tool_portal.env.example config/g1d_tool_portal.env
```

必须按机器人填写盒子地址。其他常用选项：

| 配置 | 说明 |
| --- | --- |
| `G1D_ARM_IK_VERSION` | 当前实际部署的机械臂软件版本，例如 `6.2`；未知时留空，页面不会猜测版本 |
| `G1D_SUCTION_TYPE` | `evs01` 或 `es80z` |
| `G1D_ES80Z_LOCATION` | ES80Z 后端运行在 `body` 或 `box` |
| `G1D_CONTROL_API_ENABLED` | 是否展示可选的 28090 聚合控制 API |

版本号与运行 feature 分开处理：版本来自设备配置或服务自身健康接口；feature 尽量来自实时状态。这样升级代码后，不会出现目录写着新版本而实际仍运行旧节点的问题。

## 只读接口

- `/`：目录页面。
- `/health`：门户自身健康状态及版本。
- `/api/catalog`：全部工具、探测结果、版本、feature 和关键运行事实。

门户只发起 HTTP GET 健康检查，不发布 ROS Topic，也不会发送机器人控制命令。

安装、手动启动、迁移和回滚见 [集成服务部署](INTEGRATED_SERVICES_DEPLOYMENT.md)。
