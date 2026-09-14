# ES80Z 吸盘代理

源码：`services/es80z_api_proxy/`；默认端口：18080；服务名：`suction-api-proxy.service`。

配置：从 `config/suction_api_proxy.env.example` 创建 `config/suction_api_proxy.env`，设置 `SUCTION_PROXY_UPSTREAM`。页面入口 `/`、`/ui`、`/index.html`；原有 API 请求路径直接转发到盒子 ES80Z 后端。此项目只集成本体代理，不包含盒子串口服务。

安装、手动启动、迁移和回滚见 [集成服务部署](INTEGRATED_SERVICES_DEPLOYMENT.md)。
