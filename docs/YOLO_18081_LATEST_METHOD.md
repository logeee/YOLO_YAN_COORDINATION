# 18081 YOLO 服务:切换到最新方法(双目深度 + mask内特征匹配)

本文说明如何把对 `18081` 端口的调用,从**原来的方法**切到**最新的方法**。

要点:**请求方式不变,变的只是读返回里的哪个字段。**

- 原来的方法:读返回顶层的 `center_xyz_mm`(左目 mono-PnP)
- 最新的方法 ⑨:读返回里的 `stereo_plane.center_xyz_mm`(双目深度 + mask内特征匹配,`method = feature_epipolar_ransac`)

---

## 1. 请求(不变)

端点和参数都和原来一样。`GET /xyz` 或 `POST /pose` 均可,服务端一次就把所有方法都算好返回。

> 唯一要确认的一点:请求里需要**同时带左右目的内参 + 畸变**(最新方法是双目算法)。如果你原来的调用已经传了 `fx_right/.../dist_coeffs_right`,那就完全不用改请求。

GET 示例:

```bash
curl "http://192.168.123.164:18081/xyz?label=Xizi_Liqun\
&fx=275.06&fy=275.39&cx=305.71&cy=268.34&dist_coeffs=0.059982,-0.071129,-0.000374,0.000152,0.017247\
&fx_right=274.30&fy_right=274.57&cx_right=289.72&cy_right=274.49&dist_coeffs_right=0.062923,-0.077175,-0.000405,-0.00007,0.019962"
```

POST 示例(等价):

```bash
curl -X POST http://192.168.123.164:18081/pose \
  -H "Content-Type: application/json" \
  -d '{
        "label": "Xizi_Liqun",
        "fx": 275.06, "fy": 275.39, "cx": 305.71, "cy": 268.34,
        "dist_coeffs": [0.059982, -0.071129, -0.000374, 0.000152, 0.017247],
        "fx_right": 274.30, "fy_right": 274.57, "cx_right": 289.72, "cy_right": 274.49,
        "dist_coeffs_right": [0.062923, -0.077175, -0.000405, -0.00007, 0.019962]
      }'
```

参数说明:

| 参数 | 含义 |
|---|---|
| `label` | 要识别的目标名(如 `Xizi_Liqun`);不传则取置信度最高的目标 |
| `fx, fy, cx, cy` | 左目内参 |
| `dist_coeffs` | 左目畸变 `k1,k2,p1,p2,k3`(query 用逗号分隔,JSON 用数组) |
| `fx_right, fy_right, cx_right, cy_right` | 右目内参 |
| `dist_coeffs_right` | 右目畸变 |

内参/畸变取自标定文件 `.../20260624134100/calibration_result.json` 的 `left_camera` / `right_camera`。

---

## 2. 返回(只改这里)

### 原来的方法

```python
resp = r.json()
x, y, z = resp["center_xyz_mm"]      # 左目 mono-PnP 结果,单位 mm
```

### 改为最新的方法 ⑨

```python
resp = r.json()
sp = resp["stereo_plane"]            # 最新方法:双目深度 + mask内特征匹配
x, y, z = sp["center_xyz_mm"]        # 单位 mm
```

就这一处改动:把读 `resp["center_xyz_mm"]` 换成读 `resp["stereo_plane"]["center_xyz_mm"]`。

坐标系和单位与原来一致:`left_camera_optical`、毫米(mm)。

---

## 3. `stereo_plane` 返回字段

| 字段 | 含义 |
|---|---|
| `available` | 本帧最新方法是否算出结果 |
| `method` | 固定 `feature_epipolar_ransac` |
| `frame` | 坐标系,`left_camera_optical` |
| `center_xyz_mm` / `x_mm` `y_mm` `z_mm` | 目标中心坐标(mm) |
| `depth_mm` | 沿光轴深度(mm) |
| `range_from_left_camera_mm` | 到左相机的直线距离(mm) |
| `direction_unit_xyz` | 中心方向单位向量 |
| `corner_xyz_mm` | 顶面四角坐标 |
| `top_plane_normal_xyz` | 顶面法向量 |
| `top_plane_camera_to_vertical_deg` | 顶面相对竖直方向夹角(度) |
| `long_axis_unit_xyz` | 长轴单位向量 |
| `num_features` / `num_matches` / `num_inliers` / `inlier_ratio` | 特征/匹配/内点数量与内点比例 |
| `plane_rms_mm` | 平面拟合 RMS(mm) |
| `epipolar_rms_px` | 极线残差 RMS(像素) |
| `inlier_depth_range_mm` | 内点深度范围(mm) |
| `baseline_mm` | 双目基线(mm) |
