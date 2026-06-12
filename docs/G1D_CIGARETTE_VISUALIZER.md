# G1-D 烟盒相对位置可视化

这是独立可视化服务，不改 YOLO 推理和微调服务。

## 启动

```bash
cd ~/YOLO_YAN_COORDINATION
bash scripts/g1d_cigarette_visualizer_server.sh
```

打开：

```text
http://127.0.0.1:18085/
http://<机器人IP>:18085/
```

服务检查：

```bash
curl -s http://127.0.0.1:18085/health
```

## 数据来源

页面默认读：

```text
http://127.0.0.1:18081/xyz
```

可以在页面里改成：

```text
http://192.168.60.121:18081/xyz
```

也可以选择 `XiongMao`、`Xizi_Liqun` 或自动类别。

## 当前模型

- 使用 `visualization/g1d_cigarette_viewer/g1_d.urdf` 解析 G1-D link/joint。
- 当前没有 `meshes/*.STL`，所以页面显示 URDF 骨架和代理体，不是真实外观 mesh。
- 左目相机在 URDF 里没有独立 link，页面提供相机挂载点 `X/Y/Z` 手动微调。
- 烟盒上表面尺寸来自 `/xyz` 的 `object_top_size_mm`，没有时使用默认值：
  - `XiongMao`: `161mm x 95mm`
  - `Xizi_Liqun`: `280mm x 89mm`
- 烟盒厚度目前没有实测值，页面默认 `20mm`，可以直接改。

## 坐标约定

可视化场景使用 G1-D 地面坐标：

```text
X = 机器人前方
Y = 机器人左方
Z = 垂直向上
```

`/xyz` 的左目光学坐标会根据现有 `robot_alignment.basis` 转到这个坐标系后显示。

## 本地测试

不连机器人时，页面会加载 `sample_pose.json`。  
连机器人时点击 `读取 /xyz`，页面会通过 `18085` 服务端代理请求 YOLO，避免浏览器跨域问题。
