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

页面启动时会先读 YOLO：

```text
http://127.0.0.1:18081/xyz
```

如果 YOLO 没开，页面会退回 `sample_pose.json`，方便本地测试。

可以在页面里改成：

```text
http://192.168.60.121:18081/xyz
```

也可以选择 `XiongMao`、`Xizi_Liqun` 或自动类别。
页面里的 `自动 YOLO` 默认打开，会按刷新间隔持续读取 `/xyz`。

## 相机

左目相机默认绑定到 URDF 的 `head_link`，再叠加页面里的相机偏移：

```text
camera = head_link + camera_offset
```

页面会画出左目 optical 坐标轴：

```text
红色：camera +X，图像右方
绿色：camera +Y，图像下方
蓝色：camera +Z，光轴方向
```

光轴 `camera +Z` 使用之前确认过的安装角 `42.4°`，相对地面垂直方向向前下方倾斜。

## 机器人状态

页面会读：

```text
/api/robot_state
```

默认返回展开立柱：

```json
{
  "column_extension_mm": 420,
  "joints": {
    "LZ_mt_Joint": 0.21,
    "LZ_it_Joint": 0.21
  }
}
```

后续如果有真实状态服务，可以让 `/api/robot_state` 返回同样格式。`joints` 里的单位是 URDF 单位：伸缩关节是米，旋转关节是弧度。前端会按 joint 名称驱动 URDF 模型变形。

## 当前模型

- 使用 Unitree 官方 `g1_d_description`：
  - `visualization/g1d_cigarette_viewer/g1_d.urdf`
  - `visualization/g1d_cigarette_viewer/meshes/*.STL`
- 页面会按 URDF 的 link/joint 和 `<visual><mesh>` 加载 STL。
- 立柱使用 URDF 里的两个 prismatic 关节：
  - `LZ_mt_Joint`: `0 ~ 210mm`
  - `LZ_it_Joint`: `0 ~ 210mm`
  - 页面默认总展开量 `420mm`，也就是两节都展开到上限。
- 左目相机在 URDF 里没有独立 link，页面提供相机挂载点 `X/Y/Z` 手动微调。
- 烟盒上表面尺寸来自 `/xyz` 的 `object_top_size_mm`，没有时使用默认值：
  - `XiongMao`: `161mm x 95mm`
  - `Xizi_Liqun`: `280mm x 89mm`
- 烟盒厚度目前没有实测值，页面默认 `20mm`，可以直接改。
- 默认视角是正常地面视角，地面水平铺开，机器人立在地面上；页面也提供俯视和侧视按钮。

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
