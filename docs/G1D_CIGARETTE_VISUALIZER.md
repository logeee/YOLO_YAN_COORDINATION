# G1-D 烟盒相对位置可视化

这是 G1-D 和烟盒相对位置的 3D 可视化。当前先作为独立页面测试，后续目标是嵌到 YOLO `/debug` 页面里。

核心原则：

- 烟盒位置、方向、距离来自 YOLO `/xyz`。
- 机器人模型姿态来自本体 DDS 状态。
- 可视化层只消费这两类数据，不重新做 YOLO 识别，也不重新算 PnP。

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

YOLO `/xyz` 里会返回一个精简对象：

```json
{
  "g1d_visualization": {
    "yolo": {},
    "box": {},
    "metrics": {},
    "camera": {},
    "robot_state": {}
  }
}
```

后续嵌入 `/debug` 页面时，3D 组件优先读取这个对象。

## 关键字段

3D 展示和后续微调主要用这些数据：

| 字段 | 来源 | 用途 |
| --- | --- | --- |
| DDS `rt/lowstate` | G1-D 本体 | 驱动腰部和手臂 URDF 关节 |
| DDS `rt/hispeed_state` | G1-D 本体 | 驱动立柱高度 |
| `g1d_visualization.metrics.turn_to_target_yaw_deg` | YOLO/PnP | 机器人朝目标中心需要转的角 |
| `g1d_visualization.metrics.box_long_axis_yaw_deg` | YOLO/PnP | 烟盒长轴相对机器人前方的角 |
| `g1d_visualization.metrics.center_vertical_down_mm` | YOLO/PnP | 地面垂直方向到烟盒上表面中心的距离 |
| `g1d_visualization.metrics.center_ground_forward_mm` | YOLO/PnP | 中心点的地面前向距离 |
| `g1d_visualization.metrics.near_edge_ground_forward_mm` | YOLO/PnP | 近端边中点的地面前向距离，底盘距离微调优先用这个 |
| `g1d_visualization.box.center_xyz_mm` | YOLO/PnP | 烟盒中心点，left camera optical 坐标 |
| `g1d_visualization.box.near_edge_midpoint_xyz_mm` | YOLO/PnP | 靠近机器人那条边的中点 |
| `g1d_visualization.camera.mount_parent_link` | 固定配置 | left camera 挂在 `torso_link` |
| `g1d_visualization.camera.camera_to_vertical_deg` | 固定配置 | left camera 光轴与地面垂直方向夹角，当前 42.4° |

## 相机

左目相机默认使用官方 G1 URDF 里的 `d435_joint` 挂点，绑定到 `torso_link`，再叠加页面里的相机局部偏移：

```text
camera_world = torso_link_world * camera_offset_in_torso_link
```

页面会画出左目 optical 坐标轴：

```text
红色：camera +X，图像右方
绿色：camera +Y，图像下方
蓝色：camera +Z，光轴方向
```

光轴 `camera +Z` 使用之前确认过的安装角 `42.4°`，相对地面垂直方向向前下方倾斜。

注意：相机偏移和 optical 坐标轴现在都会跟随 `torso_link` 的 world transform。也就是说立柱、腰部和上身父链变化后，left camera 会跟着机器人模型一起移动和旋转。

当前使用 `unitreerobotics/unitree_ros` 的 `robots/g1_with_brainco_hand/g1_29dof_mode_15_brainco_hand.urdf`：

```xml
<joint name="d435_joint" type="fixed">
  <origin xyz="0.0576235 0.01753 0.42987" rpy="0 0.8307767239493009 0"/>
  <parent link="torso_link"/>
  <child link="d435_link"/>
</joint>
```

G1 URDF 只给了整体 `d435_link`，没有单独给 left/right imager link；页面当前用这个官方 d435 挂点作为 left camera 的可视化基准点。

## 机器人状态

页面会读：

```text
/api/robot_state
```

机器人上默认优先读宇树 DDS：

```text
rt/lowstate        unitree_hg LowState，读取 motor_state[i].q
rt/hispeed_state   geometry_msgs Point32，读取 y 作为立柱当前高度
```

服务启动参数：

```bash
bash scripts/g1d_cigarette_visualizer_server.sh
```

默认环境变量：

```text
PYTHON=/home/unitree/miniconda3/envs/tv/bin/python
UNITREE_SDK2PY_PATH=/home/unitree/unitree_sdk2_python
DDS_INTERFACE=eth0
DDS_LOWSTATE_TOPIC=rt/lowstate
DDS_HISPEED_TOPIC=rt/hispeed_state
```

如果 DDS 不可用，才返回默认展开立柱：

```json
{
  "column_extension_mm": 420,
  "joints": {
    "LZ_mt_Joint": 0.21,
    "LZ_it_Joint": 0.21
  }
}
```

`joints` 里的单位是 URDF 单位：伸缩关节是米，旋转关节是弧度。前端会按 joint 名称驱动 URDF 模型变形。

DDS 到 URDF 的当前映射：

```text
lowstate[12] -> torso_Joint
lowstate[14] -> Yaw_Joint
lowstate[15:22] -> left shoulder/elbow/wrist
lowstate[22:29] -> right shoulder/elbow/wrist
hispeed_state.y -> LZ_mt_Joint + LZ_it_Joint
```

仍然兼容外部状态服务或 ROS `JointState` 风格：

```json
{
  "joint_states": {
    "name": ["LZ_mt_Joint", "LZ_it_Joint"],
    "position": [0.21, 0.21]
  }
}
```

## 当前模型

- 使用 Unitree 官方 `g1_d_description`：
  - `visualization/g1d_cigarette_viewer/g1_d.urdf`
  - `visualization/g1d_cigarette_viewer/meshes/*.STL`
- 页面会按 URDF 的 link/joint 和 `<visual><mesh>` 加载 STL。
- 立柱使用 URDF 里的两个 prismatic 关节：
  - `LZ_mt_Joint`: `0 ~ 210mm`
  - `LZ_it_Joint`: `0 ~ 210mm`
  - 页面默认总展开量 `420mm`，也就是两节都展开到上限。
- G1 URDF 里有 `d435_joint`，页面默认使用 `torso_link` 局部坐标 `[0.0576235, 0.01753, 0.42987]m`。G1 URDF 没有单独 left/right imager link，页面仍保留 `X/Y/Z` 输入用于现场微调。
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
