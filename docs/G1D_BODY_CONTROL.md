# G1-D 本体原子动作服务

源码：`services/g1d_body_control/node.py`；服务名：`g1d-body-control.service`。不占 HTTP 端口。

配置：从 `config/g1d_body_control.env.example` 创建 `config/g1d_body_control.env`，核对 ROS2 环境、DDS 网卡、SDK 可执行文件路径和立柱状态接口。

节点名 `g1d_body_control_api`；默认订阅 `/g1d_body_control/task_command`，发布 `/g1d_body_control/task_status`，消息类型 `std_msgs/msg/String`。沿用机器人现有代码，保留底盘、立柱任务执行及停止逻辑。此节点会真实调用 SDK，不是预览服务。

安装、手动启动、迁移和回滚见 [集成服务部署](INTEGRATED_SERVICES_DEPLOYMENT.md)。
