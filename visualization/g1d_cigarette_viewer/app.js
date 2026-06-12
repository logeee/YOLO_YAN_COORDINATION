import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const CIGARETTE_SIZES_M = {
  XiongMao: { long: 0.161, short: 0.095, thickness: 0.02 },
  Xizi_Liqun: { long: 0.280, short: 0.089, thickness: 0.02 },
  Liqun: { long: 0.280, short: 0.089, thickness: 0.02 },
};

const DEFAULT_CAMERA_TO_VERTICAL_DEG = 42.4;

window.__g1dVisualizerError = null;
window.__g1dVisualizerState = {};
window.addEventListener("error", (event) => {
  window.__g1dVisualizerError = event.message || String(event.error || event);
});
window.addEventListener("unhandledrejection", (event) => {
  window.__g1dVisualizerError = event.reason?.message || String(event.reason || event);
});

const dom = {
  viewport: document.querySelector("#viewport"),
  statusText: document.querySelector("#statusText"),
  sourceUrl: document.querySelector("#sourceUrl"),
  labelSelect: document.querySelector("#labelSelect"),
  thicknessMm: document.querySelector("#thicknessMm"),
  cameraX: document.querySelector("#cameraX"),
  cameraY: document.querySelector("#cameraY"),
  cameraZ: document.querySelector("#cameraZ"),
  fetchBtn: document.querySelector("#fetchBtn"),
  sampleBtn: document.querySelector("#sampleBtn"),
  applyBtn: document.querySelector("#applyBtn"),
  jsonInput: document.querySelector("#jsonInput"),
  metricLabel: document.querySelector("#metricLabel"),
  metricForward: document.querySelector("#metricForward"),
  metricNear: document.querySelector("#metricNear"),
  metricYaw: document.querySelector("#metricYaw"),
  sceneState: document.querySelector("#sceneState"),
};

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0c1014);

const camera = new THREE.PerspectiveCamera(48, 1, 0.01, 20);
camera.position.set(1.45, -1.8, 1.1);
camera.lookAt(0.22, 0.0, 0.45);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
dom.viewport.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0.2, 0.0, 0.35);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.48;

const root = new THREE.Group();
scene.add(root);

const robotGroup = new THREE.Group();
const cigaretteGroup = new THREE.Group();
const cameraMarkerGroup = new THREE.Group();
root.add(robotGroup, cigaretteGroup, cameraMarkerGroup);

const materials = {
  robot: new THREE.MeshStandardMaterial({ color: 0x9fb8c8, roughness: 0.75, metalness: 0.05 }),
  robotDark: new THREE.MeshStandardMaterial({ color: 0x4d6574, roughness: 0.8, metalness: 0.02 }),
  joint: new THREE.MeshStandardMaterial({ color: 0xf6c85f, roughness: 0.55 }),
  rod: new THREE.MeshStandardMaterial({ color: 0x66879b, roughness: 0.65 }),
  camera: new THREE.MeshStandardMaterial({ color: 0x64b6d9, roughness: 0.45 }),
  box: new THREE.MeshStandardMaterial({ color: 0xd8aa39, roughness: 0.58 }),
  boxTop: new THREE.MeshStandardMaterial({ color: 0x28323b, roughness: 0.85 }),
  line: new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.62 }),
};

scene.add(new THREE.HemisphereLight(0xe8f7ff, 0x2b3440, 1.7));
const keyLight = new THREE.DirectionalLight(0xffffff, 1.1);
keyLight.position.set(1.4, -1.3, 2.2);
scene.add(keyLight);

addGround();
addRobotAxes(root, 0.38);

let currentPose = null;

init();

async function init() {
  bindEvents();
  await loadRobot();
  await loadSample();
  animate();
}

function bindEvents() {
  dom.fetchBtn.addEventListener("click", fetchPose);
  dom.sampleBtn.addEventListener("click", loadSample);
  dom.applyBtn.addEventListener("click", applyJsonFromInput);
  for (const input of [dom.thicknessMm, dom.cameraX, dom.cameraY, dom.cameraZ]) {
    input.addEventListener("input", () => {
      updateCameraMarker();
      if (currentPose) renderPose(currentPose);
    });
  }

  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(dom.viewport);
  resize();
}

async function loadRobot() {
  setStatus("读取 URDF");
  const text = await fetch("./g1_d.urdf").then((res) => res.text());
  const stats = buildUrdfSkeleton(text, robotGroup);
  setStatus(`URDF ${stats.links} links，代理模型`);
  updateCameraMarker();
}

async function loadSample() {
  const pose = await fetch("./sample_pose.json").then((res) => res.json());
  applyPose(pose, "已加载示例");
}

async function fetchPose() {
  const url = new URL("/api/xyz", window.location.origin);
  url.searchParams.set("url", dom.sourceUrl.value.trim());
  if (dom.labelSelect.value) {
    url.searchParams.set("label", dom.labelSelect.value);
  }
  setStatus("读取 /xyz");
  try {
    const payload = await fetch(url).then((res) => res.json());
    if (!payload.ok) {
      throw new Error(payload.error || "读取失败");
    }
    applyPose(payload.pose, "已更新");
  } catch (error) {
    setStatus(`失败：${error.message}`);
  }
}

function applyJsonFromInput() {
  try {
    const pose = JSON.parse(dom.jsonInput.value);
    applyPose(pose, "已应用 JSON");
  } catch (error) {
    setStatus(`JSON 错误：${error.message}`);
  }
}

function applyPose(pose, status) {
  currentPose = pose;
  dom.jsonInput.value = JSON.stringify(pose, null, 2);
  renderPose(pose);
  updateMetrics(pose);
  setStatus(status);
}

function renderPose(pose) {
  cigaretteGroup.clear();
  updateCameraMarker();

  const center = getPoint(pose.center_xyz_mm || pose.near_edge_midpoint_xyz_mm);
  if (!center) {
    setStatus("缺少 center_xyz_mm");
    return;
  }

  const cameraMount = getCameraMount();
  const centerRobot = opticalPointToRobot(center, pose).add(cameraMount);
  const dims = getBoxDimensions(pose);
  const yaw = getBoxYawRobotRad(pose);

  const boxGroup = new THREE.Group();
  boxGroup.position.copy(centerRobot);
  boxGroup.rotation.z = yaw;

  const body = new THREE.Mesh(
    new THREE.BoxGeometry(dims.long, dims.short, dims.thickness),
    materials.box,
  );
  body.position.z = -dims.thickness / 2;
  boxGroup.add(body);

  const top = new THREE.Mesh(
    new THREE.BoxGeometry(dims.long * 1.002, dims.short * 1.002, 0.003),
    materials.boxTop,
  );
  top.position.z = 0.002;
  boxGroup.add(top);

  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.BoxGeometry(dims.long, dims.short, dims.thickness)),
    new THREE.LineBasicMaterial({ color: 0xf9e08c }),
  );
  edges.position.z = -dims.thickness / 2;
  boxGroup.add(edges);

  addLocalArrow(boxGroup, new THREE.Vector3(0, 0, 0.016), new THREE.Vector3(dims.long * 0.52, 0, 0), 0xf25f5c);
  addLabel(boxGroup, "head", new THREE.Vector3(dims.long * 0.55, 0, 0.055), "#f25f5c");
  addLabel(boxGroup, getPoseLabel(pose), new THREE.Vector3(0, 0, 0.075), "#f7d774");

  cigaretteGroup.add(boxGroup);
  addPointMarker(cigaretteGroup, centerRobot, 0xff4fd8, "center");

  const near = getPoint(pose.near_edge_midpoint_xyz_mm);
  if (near) {
    const nearRobot = opticalPointToRobot(near, pose).add(cameraMount);
    addPointMarker(cigaretteGroup, nearRobot, 0x35d477, "near");
  }

  const lineGeometry = new THREE.BufferGeometry().setFromPoints([cameraMount, centerRobot]);
  cigaretteGroup.add(new THREE.Line(lineGeometry, materials.line));

  const distance = cameraMount.distanceTo(centerRobot);
  addLabel(cigaretteGroup, `${Math.round(distance * 1000)} mm`, centerRobot.clone().lerp(cameraMount, 0.5).add(new THREE.Vector3(0, 0, 0.05)), "#ffffff");
  window.__g1dVisualizerState.cigaretteObjects = countObjects(cigaretteGroup);
  window.__g1dVisualizerState.lastCenterRobotM = [centerRobot.x, centerRobot.y, centerRobot.z];
  writeSceneState();

  frameScene(centerRobot);
}

function updateMetrics(pose) {
  dom.metricLabel.textContent = getPoseLabel(pose);
  const centerForward = pose.robot_alignment?.target?.ground_forward_mm;
  const nearForward = pose.near_edge_robot_alignment?.target?.ground_forward_mm;
  const yaw = pose.robot_alignment?.control_hint?.box_parallel_yaw_deg;
  dom.metricForward.textContent = formatMm(centerForward);
  dom.metricNear.textContent = formatMm(nearForward);
  dom.metricYaw.textContent = yaw == null ? "-" : `${Number(yaw).toFixed(1)}°`;
}

function buildUrdfSkeleton(urdfText, parentGroup) {
  parentGroup.clear();
  const xml = new DOMParser().parseFromString(urdfText, "application/xml");
  const links = [...xml.querySelectorAll("link")].map((node) => node.getAttribute("name")).filter(Boolean);
  const joints = [...xml.querySelectorAll("joint")].map(parseJoint).filter((joint) => joint.parent && joint.child);
  const childLinks = new Set(joints.map((joint) => joint.child));
  const rootLink = links.find((link) => !childLinks.has(link)) || links[0];
  const byParent = new Map();
  for (const joint of joints) {
    if (!byParent.has(joint.parent)) byParent.set(joint.parent, []);
    byParent.get(joint.parent).push(joint);
  }

  const rootLinkGroup = new THREE.Group();
  rootLinkGroup.name = rootLink;
  parentGroup.add(rootLinkGroup);
  addLinkProxy(rootLinkGroup, rootLink);

  function visit(linkName, group) {
    const children = byParent.get(linkName) || [];
    for (const joint of children) {
      const childGroup = new THREE.Group();
      childGroup.name = joint.child;
      childGroup.position.copy(joint.xyz);
      childGroup.rotation.set(joint.rpy.x, joint.rpy.y, joint.rpy.z, "XYZ");
      group.add(childGroup);
      addRod(group, new THREE.Vector3(), joint.xyz);
      addJointMarker(group, joint.xyz, joint.type);
      addLinkProxy(childGroup, joint.child);
      visit(joint.child, childGroup);
    }
  }

  visit(rootLink, rootLinkGroup);
  window.__g1dVisualizerState.robotObjects = countObjects(parentGroup);
  writeSceneState();
  return { links: links.length, joints: joints.length };
}

function parseJoint(node) {
  const origin = node.querySelector("origin");
  const xyz = parseVector(origin?.getAttribute("xyz") || "0 0 0");
  const rpy = parseVector(origin?.getAttribute("rpy") || "0 0 0");
  return {
    name: node.getAttribute("name"),
    type: node.getAttribute("type") || "fixed",
    parent: node.querySelector("parent")?.getAttribute("link"),
    child: node.querySelector("child")?.getAttribute("link"),
    xyz,
    rpy: { x: rpy.x, y: rpy.y, z: rpy.z },
  };
}

function parseVector(text) {
  const parts = String(text).trim().split(/\s+/).map(Number);
  return new THREE.Vector3(parts[0] || 0, parts[1] || 0, parts[2] || 0);
}

function addLinkProxy(group, name) {
  const lower = name.toLowerCase();
  let mesh;
  if (lower === "agv_link") {
    mesh = new THREE.Mesh(new THREE.BoxGeometry(0.58, 0.46, 0.12), materials.robotDark);
    mesh.position.z = 0.035;
  } else if (lower.includes("wheel")) {
    mesh = new THREE.Mesh(new THREE.CylinderGeometry(0.075, 0.075, 0.052, 32), materials.robotDark);
    mesh.rotation.x = Math.PI / 2;
  } else if (lower.includes("lz_")) {
    mesh = new THREE.Mesh(new THREE.CylinderGeometry(0.065, 0.065, 0.12, 20), materials.robot);
    mesh.position.z = 0.06;
  } else if (lower.includes("torso")) {
    mesh = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.16, 0.24), materials.robot);
    mesh.position.z = 0.04;
  } else if (lower.includes("head")) {
    mesh = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.14, 0.1), materials.robot);
    mesh.position.x = 0.035;
    mesh.position.z = 0.03;
  } else if (lower.includes("shoulder") || lower.includes("elbow") || lower.includes("wrist")) {
    mesh = new THREE.Mesh(new THREE.SphereGeometry(0.025, 16, 12), materials.robot);
  } else {
    mesh = new THREE.Mesh(new THREE.SphereGeometry(0.012, 12, 8), materials.rod);
  }
  mesh.name = `${name}_proxy`;
  group.add(mesh);
}

function addRod(group, start, end) {
  const length = start.distanceTo(end);
  if (length < 0.006) return;
  const geometry = new THREE.CylinderGeometry(0.008, 0.008, length, 10);
  const mesh = new THREE.Mesh(geometry, materials.rod);
  const mid = start.clone().lerp(end, 0.5);
  const direction = end.clone().sub(start).normalize();
  mesh.position.copy(mid);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
  group.add(mesh);
}

function addJointMarker(group, position, type) {
  const radius = type === "fixed" ? 0.012 : 0.018;
  const marker = new THREE.Mesh(new THREE.SphereGeometry(radius, 14, 10), materials.joint);
  marker.position.copy(position);
  group.add(marker);
}

function addGround() {
  const grid = new THREE.GridHelper(3.2, 32, 0x3b4a54, 0x202a31);
  grid.rotation.x = Math.PI / 2;
  grid.position.z = -0.055;
  scene.add(grid);

  const plane = new THREE.Mesh(
    new THREE.PlaneGeometry(3.2, 3.2),
    new THREE.MeshStandardMaterial({ color: 0x0e151a, roughness: 0.9, metalness: 0.0 }),
  );
  plane.position.z = -0.058;
  scene.add(plane);
}

function addRobotAxes(group, length) {
  const origin = new THREE.Vector3(0, 0, 0.02);
  group.add(new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), origin, length, 0xff5a5f, 0.06, 0.035));
  group.add(new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), origin, length, 0x35d477, 0.06, 0.035));
  group.add(new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), origin, length, 0x4f8cff, 0.06, 0.035));
  addLabel(group, "X 前", new THREE.Vector3(length + 0.03, 0, 0.04), "#ff777b");
  addLabel(group, "Y 左", new THREE.Vector3(0, length + 0.03, 0.04), "#53e28d");
  addLabel(group, "Z 上", new THREE.Vector3(0, 0, length + 0.05), "#72a3ff");
}

function updateCameraMarker() {
  cameraMarkerGroup.clear();
  const mount = getCameraMount();
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.055, 0.035, 0.035), materials.camera);
  body.position.copy(mount);
  cameraMarkerGroup.add(body);
  addLabel(cameraMarkerGroup, "left camera", mount.clone().add(new THREE.Vector3(0.02, 0.02, 0.055)), "#9be2ff");
  cameraMarkerGroup.add(new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), mount, 0.16, 0xff5a5f, 0.035, 0.02));
}

function getCameraMount() {
  return new THREE.Vector3(
    Number(dom.cameraX.value || 0),
    Number(dom.cameraY.value || 0),
    Number(dom.cameraZ.value || 0),
  );
}

function getPoint(value) {
  if (!Array.isArray(value) || value.length < 3) return null;
  return new THREE.Vector3(Number(value[0]), Number(value[1]), Number(value[2]));
}

function opticalPointToRobot(pointMm, pose) {
  const basis = getOpticalBasis(pose);
  const right = dot(pointMm, basis.xRight);
  const forward = dot(pointMm, basis.groundForward);
  const verticalDown = dot(pointMm, basis.verticalDown);
  return new THREE.Vector3(forward / 1000, -right / 1000, -verticalDown / 1000);
}

function opticalVectorToRobot(vector, pose) {
  const basis = getOpticalBasis(pose);
  const right = dot(vector, basis.xRight);
  const forward = dot(vector, basis.groundForward);
  const verticalDown = dot(vector, basis.verticalDown);
  return new THREE.Vector3(forward, -right, -verticalDown).normalize();
}

function getOpticalBasis(pose) {
  const basis = pose.robot_alignment?.basis;
  if (basis?.x_right_unit_xyz && basis?.ground_forward_unit_xyz && basis?.vertical_down_unit_xyz) {
    return {
      xRight: new THREE.Vector3(...basis.x_right_unit_xyz),
      groundForward: new THREE.Vector3(...basis.ground_forward_unit_xyz),
      verticalDown: new THREE.Vector3(...basis.vertical_down_unit_xyz),
    };
  }
  const theta = THREE.MathUtils.degToRad(Number(pose.top_plane_camera_to_vertical_deg || DEFAULT_CAMERA_TO_VERTICAL_DEG));
  return {
    xRight: new THREE.Vector3(1, 0, 0),
    groundForward: new THREE.Vector3(0, -Math.cos(theta), Math.sin(theta)),
    verticalDown: new THREE.Vector3(0, Math.sin(theta), Math.cos(theta)),
  };
}

function getBoxDimensions(pose) {
  const label = getPoseLabel(pose);
  const catalog = CIGARETTE_SIZES_M[label] || CIGARETTE_SIZES_M.Xizi_Liqun;
  let longSide = catalog.long;
  let shortSide = catalog.short;
  if (Array.isArray(pose.object_top_size_mm) && pose.object_top_size_mm.length >= 2) {
    const a = Number(pose.object_top_size_mm[0]) / 1000;
    const b = Number(pose.object_top_size_mm[1]) / 1000;
    longSide = Math.max(a, b);
    shortSide = Math.min(a, b);
  }
  const thickness = Math.max(0.005, Number(dom.thicknessMm.value || catalog.thickness * 1000) / 1000);
  return { long: longSide, short: shortSide, thickness };
}

function getBoxYawRobotRad(pose) {
  const axis = getPoint(pose.box_head_point?.head_to_tail_unit_xyz);
  if (axis) {
    const robotAxis = opticalVectorToRobot(axis, pose);
    return Math.atan2(robotAxis.y, robotAxis.x);
  }
  const axisYawRightDeg = pose.robot_alignment?.box_axis?.axis_yaw_head_to_tail_deg;
  if (axisYawRightDeg != null) {
    return THREE.MathUtils.degToRad(-Number(axisYawRightDeg));
  }
  const commandYawDeg = pose.robot_alignment?.control_hint?.box_parallel_yaw_deg;
  if (commandYawDeg != null) {
    return THREE.MathUtils.degToRad(Number(commandYawDeg));
  }
  return 0;
}

function getPoseLabel(pose) {
  return pose.selected_yolo_label || pose.requested_yolo_label || dom.labelSelect.value || "Xizi_Liqun";
}

function addLocalArrow(group, start, vector, color) {
  const direction = vector.clone().normalize();
  group.add(new THREE.ArrowHelper(direction, start, vector.length(), color, 0.035, 0.02));
}

function addPointMarker(group, position, color, label) {
  const marker = new THREE.Mesh(
    new THREE.SphereGeometry(0.018, 18, 12),
    new THREE.MeshStandardMaterial({ color, roughness: 0.45 }),
  );
  marker.position.copy(position);
  group.add(marker);
  addLabel(group, label, position.clone().add(new THREE.Vector3(0.025, 0.015, 0.03)), `#${color.toString(16).padStart(6, "0")}`);
}

function addLabel(group, text, position, color = "#ffffff") {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = 256 * ratio;
  canvas.height = 64 * ratio;
  ctx.scale(ratio, ratio);
  ctx.font = "700 22px Segoe UI, Microsoft YaHei, sans-serif";
  ctx.textBaseline = "middle";
  ctx.lineWidth = 5;
  ctx.strokeStyle = "rgba(0, 0, 0, 0.72)";
  ctx.fillStyle = color;
  ctx.strokeText(text, 8, 32);
  ctx.fillText(text, 8, 32);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true }));
  sprite.position.copy(position);
  sprite.scale.set(0.26, 0.065, 1);
  group.add(sprite);
  return sprite;
}

function frameScene(target) {
  controls.target.lerp(target, 0.35);
}

function resize() {
  const rect = dom.viewport.getBoundingClientRect();
  const width = Math.max(1, rect.width);
  const height = Math.max(1, rect.height);
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

function animate() {
  renderer.setAnimationLoop(() => {
    controls.update();
    renderer.render(scene, camera);
  });
}

function setStatus(text) {
  dom.statusText.textContent = text;
}

function formatMm(value) {
  return value == null ? "-" : `${Number(value).toFixed(1)} mm`;
}

function dot(left, right) {
  return left.x * right.x + left.y * right.y + left.z * right.z;
}

function countObjects(object) {
  let count = 0;
  object.traverse(() => {
    count += 1;
  });
  return count;
}

function writeSceneState() {
  if (!dom.sceneState) return;
  dom.sceneState.value = JSON.stringify({
    error: window.__g1dVisualizerError,
    robotObjects: window.__g1dVisualizerState.robotObjects || 0,
    cigaretteObjects: window.__g1dVisualizerState.cigaretteObjects || 0,
    lastCenterRobotM: window.__g1dVisualizerState.lastCenterRobotM || null,
  });
}
