import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";

const CIGARETTE_SIZES_M = {
  XiongMao: { long: 0.161, short: 0.095, thickness: 0.02 },
  Xizi_Liqun: { long: 0.280, short: 0.089, thickness: 0.02 },
  Liqun: { long: 0.280, short: 0.089, thickness: 0.02 },
};

const DEFAULT_CAMERA_TO_VERTICAL_DEG = 42.4;
const COLUMN_JOINT_NAMES = ["LZ_mt_Joint", "LZ_it_Joint"];
const DEFAULT_COLUMN_EXTENSION_MM = 420;
const DEFAULT_CAMERA_OFFSET_M = new THREE.Vector3(0.08, 0.04, 0.43);

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
  robotStateUrl: document.querySelector("#robotStateUrl"),
  labelSelect: document.querySelector("#labelSelect"),
  thicknessMm: document.querySelector("#thicknessMm"),
  columnExtensionMm: document.querySelector("#columnExtensionMm"),
  cameraX: document.querySelector("#cameraX"),
  cameraY: document.querySelector("#cameraY"),
  cameraZ: document.querySelector("#cameraZ"),
  fetchBtn: document.querySelector("#fetchBtn"),
  stateBtn: document.querySelector("#stateBtn"),
  autoYolo: document.querySelector("#autoYolo"),
  refreshSec: document.querySelector("#refreshSec"),
  sampleBtn: document.querySelector("#sampleBtn"),
  applyBtn: document.querySelector("#applyBtn"),
  normalViewBtn: document.querySelector("#normalViewBtn"),
  topViewBtn: document.querySelector("#topViewBtn"),
  sideViewBtn: document.querySelector("#sideViewBtn"),
  jsonInput: document.querySelector("#jsonInput"),
  metricLabel: document.querySelector("#metricLabel"),
  metricForward: document.querySelector("#metricForward"),
  metricVertical: document.querySelector("#metricVertical"),
  metricNear: document.querySelector("#metricNear"),
  metricTurn: document.querySelector("#metricTurn"),
  metricYaw: document.querySelector("#metricYaw"),
  metricColumn: document.querySelector("#metricColumn"),
  metricState: document.querySelector("#metricState"),
  sceneState: document.querySelector("#sceneState"),
};

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0c1014);

const camera = new THREE.PerspectiveCamera(48, 1, 0.01, 20);
camera.position.set(1.35, -1.35, 0.92);
camera.up.set(0, 0, 1);
camera.lookAt(0.28, 0.0, 0.24);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
dom.viewport.appendChild(renderer.domElement);

const stlLoader = new STLLoader();

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0.28, 0.0, 0.24);
controls.enableDamping = true;
controls.enableRotate = true;
controls.minDistance = 0.45;
controls.maxDistance = 5.0;

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
  urdfBody: new THREE.MeshStandardMaterial({ color: 0xd4e4ef, roughness: 0.7, metalness: 0.04 }),
  urdfBodyDark: new THREE.MeshStandardMaterial({ color: 0x41525d, roughness: 0.82, metalness: 0.05 }),
  urdfArm: new THREE.MeshStandardMaterial({ color: 0x86aabd, roughness: 0.72, metalness: 0.03 }),
  camera: new THREE.MeshStandardMaterial({ color: 0x64b6d9, roughness: 0.45 }),
  box: new THREE.MeshStandardMaterial({ color: 0xd8aa39, roughness: 0.58 }),
  boxTop: new THREE.MeshStandardMaterial({ color: 0x28323b, roughness: 0.85 }),
  line: new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.62 }),
};
const urdfMaterialCache = new Map();

scene.add(new THREE.HemisphereLight(0xe8f7ff, 0x2b3440, 1.7));
const keyLight = new THREE.DirectionalLight(0xffffff, 1.1);
keyLight.position.set(1.4, -1.3, 2.2);
scene.add(keyLight);

addGround();
addRobotAxes(root, 0.38);

let currentPose = null;
let currentRobotState = null;
let currentView = "normal";
let lastFocus = new THREE.Vector3(0.24, 0.0, 0.0);
let urdfJointControls = new Map();
let urdfLinkGroups = new Map();
let autoFetchTimer = null;

init();

async function init() {
  bindEvents();
  await loadRobot();
  await fetchRobotState({ silent: true });
  await fetchPose({ fallbackToSample: true });
  setupAutoRefresh();
  animate();
}

function bindEvents() {
  dom.fetchBtn.addEventListener("click", () => fetchPose({ fallbackToSample: false }));
  dom.stateBtn.addEventListener("click", () => fetchRobotState({ silent: false }));
  dom.autoYolo.addEventListener("change", setupAutoRefresh);
  dom.refreshSec.addEventListener("input", setupAutoRefresh);
  dom.sampleBtn.addEventListener("click", loadSample);
  dom.applyBtn.addEventListener("click", applyJsonFromInput);
  dom.normalViewBtn.addEventListener("click", () => setView("normal"));
  dom.topViewBtn.addEventListener("click", () => setView("top"));
  dom.sideViewBtn.addEventListener("click", () => setView("side"));
  dom.columnExtensionMm.addEventListener("input", () => {
    applyColumnExtension();
    if (currentPose) renderPose(currentPose);
  });
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

function setupAutoRefresh() {
  if (autoFetchTimer) {
    window.clearInterval(autoFetchTimer);
    autoFetchTimer = null;
  }
  if (!dom.autoYolo.checked) return;
  const intervalMs = Math.max(500, Number(dom.refreshSec.value || 1.5) * 1000);
  autoFetchTimer = window.setInterval(() => {
    fetchRobotState({ silent: true }).finally(() => {
      fetchPose({ fallbackToSample: false, silent: true });
    });
  }, intervalMs);
}

async function loadRobot() {
  setStatus("读取 URDF");
  const text = await fetch("./g1_d.urdf").then((res) => res.text());
  const stats = buildUrdfSkeleton(text, robotGroup);
  setStatus(`URDF ${stats.links} links，${stats.meshVisuals} 个 STL visual`);
  updateCameraMarker();
}

async function loadSample() {
  const pose = await fetch("./sample_pose.json").then((res) => res.json());
  applyPose(pose, "已加载示例");
}

async function fetchPose({ fallbackToSample = false, silent = false } = {}) {
  const url = new URL("/api/xyz", window.location.origin);
  url.searchParams.set("url", dom.sourceUrl.value.trim());
  if (dom.labelSelect.value) {
    url.searchParams.set("label", dom.labelSelect.value);
  }
  if (!silent) setStatus("读取 YOLO /xyz");
  try {
    const payload = await fetch(url).then((res) => res.json());
    if (!payload.ok) {
      throw new Error(payload.error || "读取失败");
    }
    applyPose(payload.pose, "YOLO 已更新");
  } catch (error) {
    if (fallbackToSample) {
      await loadSample();
      setStatus(`YOLO 失败，已加载示例：${error.message}`);
    } else if (!silent) {
      setStatus(`YOLO 失败：${error.message}`);
    }
  }
}

async function fetchRobotState({ silent = false } = {}) {
  const rawUrl = dom.robotStateUrl.value.trim() || "/api/robot_state";
  const url = new URL(rawUrl, window.location.origin);
  if (!silent) setStatus("读取机器人状态");
  try {
    const payload = await fetch(url).then((res) => res.json());
    if (!payload.ok) {
      throw new Error(payload.error || "读取状态失败");
    }
    applyRobotState(payload.state || payload);
    if (!silent) setStatus("机器人状态已更新");
  } catch (error) {
    if (!silent) setStatus(`状态失败：${error.message}`);
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

  const cameraFrame = getCameraFrame(pose);
  const cameraMount = cameraFrame.origin;
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
  const vizMetrics = pose.g1d_visualization?.metrics || {};
  dom.metricLabel.textContent = getPoseLabel(pose);
  const centerForward = vizMetrics.center_ground_forward_mm ?? pose.robot_alignment?.target?.ground_forward_mm;
  const centerVertical = vizMetrics.center_vertical_down_mm ?? pose.robot_alignment?.target?.vertical_down_mm;
  const nearForward = vizMetrics.near_edge_ground_forward_mm ?? pose.near_edge_robot_alignment?.target?.ground_forward_mm;
  const turn = vizMetrics.turn_to_target_yaw_deg ?? pose.robot_alignment?.control_hint?.turn_first_yaw_deg;
  const yaw = vizMetrics.box_long_axis_yaw_deg ?? pose.robot_alignment?.control_hint?.box_parallel_yaw_deg;
  dom.metricForward.textContent = formatMm(centerForward);
  dom.metricVertical.textContent = formatMm(centerVertical);
  dom.metricNear.textContent = formatMm(nearForward);
  dom.metricTurn.textContent = formatDeg(turn);
  dom.metricYaw.textContent = yaw == null ? "-" : `${Number(yaw).toFixed(1)}°`;
  dom.metricColumn.textContent = `${Math.round(getColumnExtensionM() * 1000)} mm`;
  dom.metricState.textContent = currentRobotState?.source || "default";
}

function applyRobotState(state) {
  currentRobotState = state || {};
  const joints = normalizeJointState(currentRobotState);
  if (currentRobotState.column_extension_mm != null) {
    dom.columnExtensionMm.value = String(Math.round(Number(currentRobotState.column_extension_mm)));
  }
  const stateJointValues = { ...joints };
  if (currentRobotState.column_extension_mm != null) {
    const totalM = clamp(Number(currentRobotState.column_extension_mm) / 1000, 0, 0.42);
    stateJointValues.LZ_mt_Joint = totalM / 2;
    stateJointValues.LZ_it_Joint = totalM / 2;
  }
  applyJointState(stateJointValues);
  updateColumnState();
  updateCameraMarker();
  if (currentPose) {
    renderPose(currentPose);
    updateMetrics(currentPose);
  }
  window.__g1dVisualizerState.robotStateSource = currentRobotState.source || "unknown";
  window.__g1dVisualizerState.robotStateUpdatedAt = currentRobotState.updated_at || null;
  writeSceneState();
}

function normalizeJointState(state) {
  if (state.joints && typeof state.joints === "object") {
    return state.joints;
  }
  const jointStates = state.joint_states;
  const names = jointStates?.name;
  const positions = jointStates?.position;
  if (!Array.isArray(names) || !Array.isArray(positions)) {
    return {};
  }
  const joints = {};
  for (let index = 0; index < Math.min(names.length, positions.length); index += 1) {
    joints[String(names[index])] = Number(positions[index]);
  }
  return joints;
}

function applyJointState(joints) {
  for (const [name, control] of urdfJointControls.entries()) {
    const value = joints && Object.prototype.hasOwnProperty.call(joints, name)
      ? Number(joints[name])
      : jointValue(control.joint);
    setJointTransform(control, value);
  }
  window.__g1dVisualizerState.appliedJointCount = joints ? Object.keys(joints).length : 0;
}

function buildUrdfSkeleton(urdfText, parentGroup) {
  parentGroup.clear();
  urdfJointControls = new Map();
  urdfLinkGroups = new Map();
  const xml = new DOMParser().parseFromString(urdfText, "application/xml");
  const links = [...xml.querySelectorAll("link")].map((node) => node.getAttribute("name")).filter(Boolean);
  const joints = [...xml.querySelectorAll("joint")].map(parseJoint).filter((joint) => joint.parent && joint.child);
  const visualMap = parseVisualMap(xml);
  const childLinks = new Set(joints.map((joint) => joint.child));
  const rootLink = links.find((link) => !childLinks.has(link)) || links[0];
  const byParent = new Map();
  for (const joint of joints) {
    if (!byParent.has(joint.parent)) byParent.set(joint.parent, []);
    byParent.get(joint.parent).push(joint);
  }
  const worldPositions = computeUrdfWorldPositions(rootLink, byParent);

  const rootLinkGroup = new THREE.Group();
  rootLinkGroup.name = rootLink;
  parentGroup.add(rootLinkGroup);
  urdfLinkGroups.set(rootLink, rootLinkGroup);
  if (!addUrdfVisuals(rootLinkGroup, rootLink, visualMap)) {
    addLinkProxy(rootLinkGroup, rootLink);
  }

  function visit(linkName, group) {
    const children = byParent.get(linkName) || [];
    for (const joint of children) {
      const childGroup = new THREE.Group();
      childGroup.name = joint.child;
      childGroup.position.copy(jointPosition(joint));
      childGroup.rotation.set(joint.rpy.x, joint.rpy.y, joint.rpy.z, "XYZ");
      group.add(childGroup);
      urdfLinkGroups.set(joint.child, childGroup);
      if (joint.type !== "fixed") {
        urdfJointControls.set(joint.name, {
          group: childGroup,
          joint,
          baseXYZ: joint.xyz.clone(),
          baseQuaternion: childGroup.quaternion.clone(),
        });
      }
      addRod(group, new THREE.Vector3(), childGroup.position);
      addJointMarker(group, childGroup.position, joint.type);
      if (!addUrdfVisuals(childGroup, joint.child, visualMap)) {
        addLinkProxy(childGroup, joint.child);
      }
      visit(joint.child, childGroup);
    }
  }

  visit(rootLink, rootLinkGroup);
  const urdfMeshVisuals = [...visualMap.values()].reduce((sum, visuals) => sum + visuals.length, 0);
  if (urdfMeshVisuals === 0) {
    addReadableG1DProxy(parentGroup, worldPositions);
  }
  window.__g1dVisualizerState.robotObjects = countObjects(parentGroup);
  window.__g1dVisualizerState.urdfLinks = links.length;
  window.__g1dVisualizerState.urdfJoints = joints.length;
  window.__g1dVisualizerState.urdfMeshVisuals = urdfMeshVisuals;
  window.__g1dVisualizerState.urdfLoadedMeshes = 0;
  window.__g1dVisualizerState.urdfFailedMeshes = 0;
  window.__g1dVisualizerState.urdfProxy = urdfMeshVisuals === 0;
  updateColumnState();
  writeSceneState();
  return { links: links.length, joints: joints.length, meshVisuals: urdfMeshVisuals };
}

function computeUrdfWorldPositions(rootLink, byParent) {
  const positions = new Map([[rootLink, new THREE.Vector3()]]);

  function visit(linkName) {
    const base = positions.get(linkName) || new THREE.Vector3();
    const children = byParent.get(linkName) || [];
    for (const joint of children) {
      const childPosition = base.clone().add(jointPosition(joint));
      positions.set(joint.child, childPosition);
      visit(joint.child);
    }
  }

  visit(rootLink);
  return positions;
}

function jointPosition(joint) {
  return joint.xyz.clone().add(joint.axis.clone().multiplyScalar(jointValue(joint)));
}

function jointValue(joint) {
  if (!COLUMN_JOINT_NAMES.includes(joint.name)) return 0;
  const totalM = getColumnExtensionM();
  const perJointM = totalM / COLUMN_JOINT_NAMES.length;
  return clamp(perJointM, joint.limit.lower, joint.limit.upper);
}

function setJointTransform(control, rawValue) {
  const joint = control.joint;
  const value = clampJointValue(joint, Number.isFinite(rawValue) ? rawValue : 0);
  if (joint.type === "prismatic") {
    control.group.position.copy(joint.xyz.clone().add(joint.axis.clone().multiplyScalar(value)));
    control.group.quaternion.copy(control.baseQuaternion);
  } else if (joint.type === "revolute" || joint.type === "continuous") {
    control.group.position.copy(control.baseXYZ);
    const axis = normalizedVector(joint.axis);
    const jointRotation = new THREE.Quaternion().setFromAxisAngle(axis, value);
    control.group.quaternion.copy(control.baseQuaternion).multiply(jointRotation);
  } else {
    control.group.position.copy(control.baseXYZ);
    control.group.quaternion.copy(control.baseQuaternion);
  }
}

function clampJointValue(joint, value) {
  if (joint.limit.upper > joint.limit.lower) {
    return clamp(value, joint.limit.lower, joint.limit.upper);
  }
  return value;
}

function getColumnExtensionM() {
  const rawMm = Number(dom.columnExtensionMm?.value || DEFAULT_COLUMN_EXTENSION_MM);
  return clamp(rawMm / 1000, 0, 0.42);
}

function applyColumnExtension() {
  for (const name of COLUMN_JOINT_NAMES) {
    const control = urdfJointControls.get(name);
    if (!control) continue;
    setJointTransform(control, jointValue(control.joint));
  }
  updateColumnState();
  writeSceneState();
}

function updateColumnState() {
  window.__g1dVisualizerState.columnExtensionMm = Math.round(getColumnExtensionM() * 1000);
  window.__g1dVisualizerState.columnJointValuesMm = Object.fromEntries(
    COLUMN_JOINT_NAMES.map((name) => {
      const control = urdfJointControls.get(name);
      return [name, control ? Math.round(jointValue(control.joint) * 1000) : null];
    }),
  );
}

function parseVisualMap(xml) {
  const visualMap = new Map();
  for (const linkNode of xml.querySelectorAll("link")) {
    const linkName = linkNode.getAttribute("name");
    if (!linkName) continue;
    const visuals = [];
    for (const visualNode of [...linkNode.children].filter((child) => child.tagName === "visual")) {
      const meshNode = visualNode.querySelector("geometry mesh");
      const filename = meshNode?.getAttribute("filename");
      if (!filename) continue;
      const origin = visualNode.querySelector("origin");
      const scaleText = meshNode.getAttribute("scale") || "1 1 1";
      const colorText = visualNode.querySelector("material color")?.getAttribute("rgba");
      visuals.push({
        filename: normalizeMeshPath(filename),
        xyz: parseVector(origin?.getAttribute("xyz") || "0 0 0"),
        rpy: parseVector(origin?.getAttribute("rpy") || "0 0 0"),
        scale: parseVector(scaleText),
        color: parseRgba(colorText),
      });
    }
    if (visuals.length > 0) {
      visualMap.set(linkName, visuals);
    }
  }
  return visualMap;
}

function normalizeMeshPath(filename) {
  const normalized = String(filename).replace(/\\/g, "/");
  const marker = "g1_d_description/";
  if (normalized.startsWith("package://")) {
    const index = normalized.indexOf(marker);
    return index >= 0 ? normalized.slice(index + marker.length) : normalized.replace("package://", "");
  }
  return normalized.replace(/^\.?\//, "");
}

function parseRgba(text) {
  if (!text) return [0.78, 0.84, 0.9, 1.0];
  const values = String(text).trim().split(/\s+/).map(Number);
  return [
    Number.isFinite(values[0]) ? values[0] : 0.78,
    Number.isFinite(values[1]) ? values[1] : 0.84,
    Number.isFinite(values[2]) ? values[2] : 0.9,
    Number.isFinite(values[3]) ? values[3] : 1.0,
  ];
}

function addUrdfVisuals(group, linkName, visualMap) {
  const visuals = visualMap.get(linkName) || [];
  if (visuals.length === 0) return false;
  for (const visual of visuals) {
    loadUrdfStl(group, linkName, visual);
  }
  return true;
}

function loadUrdfStl(group, linkName, visual) {
  const material = materialForRgba(visual.color);
  stlLoader.load(
    visual.filename,
    (geometry) => {
      geometry.computeVertexNormals();
      const mesh = new THREE.Mesh(geometry, material);
      mesh.name = `${linkName}_mesh`;
      mesh.position.copy(visual.xyz);
      mesh.rotation.set(visual.rpy.x, visual.rpy.y, visual.rpy.z, "XYZ");
      mesh.scale.set(visual.scale.x || 1, visual.scale.y || 1, visual.scale.z || 1);
      group.add(mesh);
      window.__g1dVisualizerState.urdfLoadedMeshes = (window.__g1dVisualizerState.urdfLoadedMeshes || 0) + 1;
      window.__g1dVisualizerState.robotObjects = countObjects(robotGroup);
      writeSceneState();
    },
    undefined,
    () => {
      addLinkProxy(group, linkName);
      window.__g1dVisualizerState.urdfFailedMeshes = (window.__g1dVisualizerState.urdfFailedMeshes || 0) + 1;
      window.__g1dVisualizerState.robotObjects = countObjects(robotGroup);
      writeSceneState();
    },
  );
}

function materialForRgba(rgba) {
  const key = rgba.map((value) => Number(value).toFixed(3)).join(",");
  if (!urdfMaterialCache.has(key)) {
    urdfMaterialCache.set(
      key,
      new THREE.MeshStandardMaterial({
        color: new THREE.Color(rgba[0], rgba[1], rgba[2]),
        opacity: rgba[3],
        transparent: rgba[3] < 0.999,
        roughness: 0.74,
        metalness: 0.04,
      }),
    );
  }
  return urdfMaterialCache.get(key);
}

function addReadableG1DProxy(parentGroup, worldPositions) {
  const proxy = new THREE.Group();
  proxy.name = "readable_urdf_proxy";
  parentGroup.add(proxy);

  addBox(proxy, new THREE.Vector3(0.0, 0.0, 0.02), [0.64, 0.50, 0.10], materials.urdfBodyDark);
  addBox(proxy, new THREE.Vector3(0.04, 0.0, 0.095), [0.40, 0.34, 0.045], materials.urdfBody);

  const leftWheel = worldPositions.get("Left_Wheel_Link") || new THREE.Vector3(0, 0.203, -0.026);
  const rightWheel = worldPositions.get("RIght_Wheel_Link") || new THREE.Vector3(0, -0.203, -0.026);
  addWheel(proxy, leftWheel);
  addWheel(proxy, rightWheel);

  const lzBase = worldPositions.get("LZ_ot_Link") || new THREE.Vector3(0, 0, 0.528);
  const torso = worldPositions.get("torso_link") || new THREE.Vector3(0.055, 0, 0.733);
  const head = worldPositions.get("head_link") || new THREE.Vector3(0.059, 0, 0.679);
  addCylinderBetween(proxy, new THREE.Vector3(lzBase.x, lzBase.y, 0.11), torso.clone().add(new THREE.Vector3(0, 0, -0.06)), 0.035, materials.urdfBody);
  addBox(proxy, torso.clone().add(new THREE.Vector3(0.02, 0, 0.03)), [0.24, 0.17, 0.20], materials.urdfBody);
  addBox(proxy, head.clone().add(new THREE.Vector3(0.05, 0, 0.035)), [0.20, 0.15, 0.10], materials.urdfBody);

  addArmProxy(proxy, worldPositions, "left");
  addArmProxy(proxy, worldPositions, "right");
  addLabel(proxy, "G1-D URDF", new THREE.Vector3(-0.18, -0.28, 0.16), "#d9ecf6");
}

function addArmProxy(proxy, worldPositions, side) {
  const prefix = `${side}_`;
  const points = [
    worldPositions.get(`${prefix}shoulder_pitch_link`),
    worldPositions.get(`${prefix}shoulder_roll_link`),
    worldPositions.get(`${prefix}shoulder_yaw_link`),
    worldPositions.get(`${prefix}elbow_link`),
    worldPositions.get(`${prefix}wrist_roll_link`),
    worldPositions.get(`${prefix}wrist_pitch_link`),
    worldPositions.get(`${prefix}wrist_yaw_link`),
  ].filter(Boolean);
  for (let index = 0; index < points.length - 1; index += 1) {
    addCylinderBetween(proxy, points[index], points[index + 1], 0.018, materials.urdfArm);
  }
  for (const point of points) {
    const marker = new THREE.Mesh(new THREE.SphereGeometry(0.028, 18, 12), materials.joint);
    marker.position.copy(point);
    proxy.add(marker);
  }
}

function parseJoint(node) {
  const origin = node.querySelector("origin");
  const axisNode = node.querySelector("axis");
  const limitNode = node.querySelector("limit");
  const xyz = parseVector(origin?.getAttribute("xyz") || "0 0 0");
  const rpy = parseVector(origin?.getAttribute("rpy") || "0 0 0");
  return {
    name: node.getAttribute("name"),
    type: node.getAttribute("type") || "fixed",
    parent: node.querySelector("parent")?.getAttribute("link"),
    child: node.querySelector("child")?.getAttribute("link"),
    xyz,
    axis: parseVector(axisNode?.getAttribute("xyz") || "0 0 0"),
    limit: {
      lower: Number(limitNode?.getAttribute("lower") || 0),
      upper: Number(limitNode?.getAttribute("upper") || 0),
    },
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

function addBox(group, position, size, material) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(size[0], size[1], size[2]), material);
  mesh.position.copy(position);
  group.add(mesh);
  const edge = new THREE.LineSegments(
    new THREE.EdgesGeometry(mesh.geometry),
    new THREE.LineBasicMaterial({ color: 0xe7f5fb, transparent: true, opacity: 0.42 }),
  );
  edge.position.copy(position);
  group.add(edge);
  return mesh;
}

function addWheel(group, position) {
  const wheel = new THREE.Mesh(new THREE.CylinderGeometry(0.078, 0.078, 0.056, 32), materials.urdfBodyDark);
  wheel.position.copy(position);
  wheel.rotation.x = Math.PI / 2;
  group.add(wheel);
  return wheel;
}

function addCylinderBetween(group, start, end, radius, material) {
  const length = start.distanceTo(end);
  if (length < 0.006) return null;
  const geometry = new THREE.CylinderGeometry(radius, radius, length, 16);
  const mesh = new THREE.Mesh(geometry, material);
  const mid = start.clone().lerp(end, 0.5);
  const direction = end.clone().sub(start).normalize();
  mesh.position.copy(mid);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
  group.add(mesh);
  return mesh;
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
  const frame = getCameraFrame(currentPose);
  const mount = frame.origin;
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.055, 0.035, 0.035), materials.camera);
  body.position.copy(mount);
  body.quaternion.copy(frame.parentQuaternion);
  cameraMarkerGroup.add(body);
  addLabel(cameraMarkerGroup, "left camera", mount.clone().add(new THREE.Vector3(0.02, 0.02, 0.055)), "#9be2ff");
  addCameraOpticalAxes(cameraMarkerGroup, frame);
}

function getCameraMount() {
  return getCameraFrame(currentPose).origin;
}

function getCameraFrame(pose = null) {
  const headTransform = getLinkWorldTransform("head_link");
  const parentPosition = headTransform?.position || new THREE.Vector3();
  const parentQuaternion = headTransform?.quaternion || new THREE.Quaternion();
  const localOffset = new THREE.Vector3(
    numberOrDefault(dom.cameraX.value, DEFAULT_CAMERA_OFFSET_M.x),
    numberOrDefault(dom.cameraY.value, DEFAULT_CAMERA_OFFSET_M.y),
    numberOrDefault(dom.cameraZ.value, DEFAULT_CAMERA_OFFSET_M.z),
  );
  const origin = parentPosition.clone().add(localOffset.clone().applyQuaternion(parentQuaternion));
  const cameraToVerticalDeg = Number(
    pose?.g1d_visualization?.camera?.camera_to_vertical_deg
      ?? pose?.top_plane_camera_to_vertical_deg
      ?? DEFAULT_CAMERA_TO_VERTICAL_DEG,
  );
  const localAxes = cameraOpticalAxesInHeadLocal(cameraToVerticalDeg);
  return {
    origin,
    parentPosition,
    parentQuaternion,
    localOffset,
    axes: {
      xRight: localAxes.xRight.clone().applyQuaternion(parentQuaternion).normalize(),
      yDown: localAxes.yDown.clone().applyQuaternion(parentQuaternion).normalize(),
      zForward: localAxes.zForward.clone().applyQuaternion(parentQuaternion).normalize(),
    },
  };
}

function getLinkWorldPosition(linkName) {
  const transform = getLinkWorldTransform(linkName);
  return transform?.position || null;
}

function getLinkWorldTransform(linkName) {
  const group = urdfLinkGroups.get(linkName);
  if (!group) return null;
  robotGroup.updateMatrixWorld(true);
  return {
    position: group.getWorldPosition(new THREE.Vector3()),
    quaternion: group.getWorldQuaternion(new THREE.Quaternion()),
  };
}

function addCameraOpticalAxes(group, frame) {
  const origin = frame.origin;
  const axes = frame.axes;
  group.add(new THREE.ArrowHelper(axes.xRight, origin, 0.16, 0xff5a5f, 0.035, 0.02));
  group.add(new THREE.ArrowHelper(axes.yDown, origin, 0.16, 0x35d477, 0.035, 0.02));
  group.add(new THREE.ArrowHelper(axes.zForward, origin, 0.22, 0x4f8cff, 0.045, 0.025));
  addLabel(group, "cam X", origin.clone().add(axes.xRight.clone().multiplyScalar(0.18)), "#ff777b");
  addLabel(group, "cam Y", origin.clone().add(axes.yDown.clone().multiplyScalar(0.18)), "#53e28d");
  addLabel(group, "cam Z 42.4°", origin.clone().add(axes.zForward.clone().multiplyScalar(0.25)), "#72a3ff");
}

function cameraOpticalAxesInHeadLocal(cameraToVerticalDeg) {
  const theta = THREE.MathUtils.degToRad(Number(cameraToVerticalDeg || DEFAULT_CAMERA_TO_VERTICAL_DEG));
  return {
    xRight: new THREE.Vector3(0, -1, 0),
    yDown: new THREE.Vector3(-Math.cos(theta), 0, -Math.sin(theta)).normalize(),
    zForward: new THREE.Vector3(Math.sin(theta), 0, -Math.cos(theta)).normalize(),
  };
}

function getPoint(value) {
  if (!Array.isArray(value) || value.length < 3) return null;
  return new THREE.Vector3(Number(value[0]), Number(value[1]), Number(value[2]));
}

function opticalPointToRobot(pointMm, pose) {
  const axes = getCameraFrame(pose).axes;
  return axes.xRight.clone().multiplyScalar(pointMm.x / 1000)
    .add(axes.yDown.clone().multiplyScalar(pointMm.y / 1000))
    .add(axes.zForward.clone().multiplyScalar(pointMm.z / 1000));
}

function opticalVectorToRobot(vector, pose) {
  const axes = getCameraFrame(pose).axes;
  return axes.xRight.clone().multiplyScalar(vector.x)
    .add(axes.yDown.clone().multiplyScalar(vector.y))
    .add(axes.zForward.clone().multiplyScalar(vector.z))
    .normalize();
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
  const commandYawDeg = pose.g1d_visualization?.metrics?.box_long_axis_yaw_deg
    ?? pose.robot_alignment?.control_hint?.box_parallel_yaw_deg;
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
  lastFocus = new THREE.Vector3(target.x * 0.5, target.y * 0.5, 0.0);
  setView(currentView, false);
}

function setView(mode, immediate = true) {
  currentView = mode;
  const focus = lastFocus.clone();
  const spread = Math.max(1.0, Math.hypot(focus.x, focus.y) * 2.4 + 0.75);
  if (mode === "top") {
    camera.up.set(1, 0, 0);
    camera.position.set(focus.x, focus.y, Math.max(1.55, spread * 1.45));
    controls.target.copy(focus);
    controls.enableRotate = false;
  } else if (mode === "side") {
    camera.up.set(0, 0, 1);
    camera.position.set(focus.x - spread * 1.25, focus.y - 0.06, 0.72);
    controls.target.set(focus.x, focus.y, 0.28);
    controls.enableRotate = true;
  } else {
    camera.up.set(0, 0, 1);
    camera.position.set(focus.x + spread * 0.88, focus.y - spread * 0.78, 0.78);
    controls.target.set(focus.x + 0.08, focus.y, 0.20);
    controls.enableRotate = true;
  }
  camera.lookAt(controls.target);
  controls.update();
  if (immediate) {
    setStatus(mode === "top" ? "俯视地面" : mode === "side" ? "侧视" : "正常地面视角");
  }
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

function formatDeg(value) {
  return value == null ? "-" : `${Number(value).toFixed(1)}°`;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function normalizedVector(vector) {
  const length = vector.length();
  if (length <= 1e-9) return new THREE.Vector3(0, 0, 1);
  return vector.clone().divideScalar(length);
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
    urdfLinks: window.__g1dVisualizerState.urdfLinks || 0,
    urdfJoints: window.__g1dVisualizerState.urdfJoints || 0,
    urdfMeshVisuals: window.__g1dVisualizerState.urdfMeshVisuals || 0,
    urdfLoadedMeshes: window.__g1dVisualizerState.urdfLoadedMeshes || 0,
    urdfFailedMeshes: window.__g1dVisualizerState.urdfFailedMeshes || 0,
    urdfProxy: Boolean(window.__g1dVisualizerState.urdfProxy),
    columnExtensionMm: window.__g1dVisualizerState.columnExtensionMm || 0,
    columnJointValuesMm: window.__g1dVisualizerState.columnJointValuesMm || {},
    cameraMountM: vectorToArray(getCameraMount()),
    cameraLocalOffsetM: vectorToArray(getCameraFrame(currentPose).localOffset),
    cameraParentLink: "head_link",
    cameraOpticalAngleDeg: DEFAULT_CAMERA_TO_VERTICAL_DEG,
    robotStateSource: window.__g1dVisualizerState.robotStateSource || null,
    robotStateUpdatedAt: window.__g1dVisualizerState.robotStateUpdatedAt || null,
    appliedJointCount: window.__g1dVisualizerState.appliedJointCount || 0,
    cigaretteObjects: window.__g1dVisualizerState.cigaretteObjects || 0,
    lastCenterRobotM: window.__g1dVisualizerState.lastCenterRobotM || null,
  });
}

function vectorToArray(vector) {
  return [roundNumber(vector.x, 4), roundNumber(vector.y, 4), roundNumber(vector.z, 4)];
}

function roundNumber(value, digits) {
  const factor = 10 ** digits;
  return Math.round(Number(value) * factor) / factor;
}

function numberOrDefault(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}
