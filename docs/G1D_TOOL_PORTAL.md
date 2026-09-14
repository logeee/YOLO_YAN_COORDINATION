# G1-D 工具门户

源码：`services/g1d_tool_portal/`；默认端口：18079；服务名：`g1d-tool-portal.service`。

配置：从 `config/g1d_tool_portal.env.example` 创建不受 Git 跟踪的 `config/g1d_tool_portal.env`，填写盒子无线和内网 IP。工具定义在 `catalog.py`；HTML、CSS、JS 在 `static/`。接口保留 `/`、`/health`、`/api/catalog`。

安装、手动启动、迁移和回滚见 [集成服务部署](INTEGRATED_SERVICES_DEPLOYMENT.md)。
