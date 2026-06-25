import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";

const CIGARETTE_SIZES_M = {
  XiongMao: { long: 0.161, short: 0.095, thickness: 0.02 },
  Xizi_Liqun: { long: 0.280, short: 0.089, thickness: 0.02 },
  Liqun: { long: 0.280, short: 0.089, thickness: 0.02 },
};

const DEFAULT_CAMERA_TO_VERTICAL_DEG = 47.6;
const COLUMN_JOINT_NAMES = ["LZ_mt_Joint", "LZ_it_Joint"];
const DEFAULT_COLUMN_EXTENSION_MM = 420;
const DEFAULT_CAMERA_PARENT_LINK = "torso_link";
const DEFAULT_CAMERA_OFFSET_M = new THREE.Vector3(0.0576235, 0.01753, 0.42987);
const URL_PARAMS = new URLSearchParams(window.location.search);
const IS_COMPACT_MODE = URL_PARAMS.get("compact") === "1" || URL_PARAMS.get("embedded") === "1";
const NO_AUTO_FETCH_POSE = URL_PARAMS.get("no_fetch") === "1" || URL_PARAMS.get("pose") === "posted";
const DEFAULT_VIEW_FOCUS = new THREE.Vector3(0.18, -0.04, 0.72);

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
  normalViewBtn: document.querySelector("#normalViewBtn"),
  topViewBtn: document.querySelector("#topViewBtn"),
  sideViewBtn: document.querySelector("#sideViewBtn"),
  metricLabel: document.querySelector("#metricLabel"),
  metricForward: document.querySelector("#metricForward"),
  metricVertical: document.querySelector("#metricVertical"),
  baseCoordOldOld: document.querySelector("#baseCoordOldOld"),
  baseCoordOldNew: document.querySelector("#baseCoordOldNew"),
  baseCoordNewNew: document.querySelector("#baseCoordNewNew"),
  baseCoordNewOld: document.querySelector("#baseCoordNewOld"),
  baseCoordOurs: document.querySelector("#baseCoordOurs"),
  baseCoordNewDist: document.querySelector("#baseCoordNewDist"),
  baseCoordDeltaEx: document.querySelector("#baseCoordDeltaEx"),
  baseCoordDeltaIn: document.querySelector("#baseCoordDeltaIn"),
  baseCoordDeltaDist: document.querySelector("#baseCoordDeltaDist"),
  baseCoordRefresh: document.querySelector("#baseCoordRefresh"),
  baseCoordStatus: document.querySelector("#baseCoordStatus"),
  useOurPnp: document.querySelector("#useOurPnp"),
  metricNear: document.querySelector("#metricNear"),
  metricTurn: document.querySelector("#metricTurn"),
  metricYaw: document.querySelector("#metricYaw"),
  metricDelta: document.querySelector("#metricDelta"),
  metricState: document.querySelector("#metricState"),
  sceneState: document.querySelector("#sceneState"),
  showOldOld: document.querySelector("#showOldOld"),
  showOldNew: document.querySelector("#showOldNew"),
  showNewNew: document.querySelector("#showNewNew"),
  showNewOld: document.querySelector("#showNewOld"),
  showNewNewDist: document.querySelector("#showNewNewDist"),
  centerOnly: document.querySelector("#centerOnly"),
  autoRefresh: document.querySelector("#autoRefresh"),
  autoInterval: document.querySelector("#autoInterval"),
};

let autoRefreshTimer = null;
let autoRefreshInFlight = false;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0c1014);

const camera = new THREE.PerspectiveCamera(48, 1, 0.01, 20);
camera.position.set(2.2, -2.1, 1.45);
camera.up.set(0, 0, 1);
camera.lookAt(DEFAULT_VIEW_FOCUS);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
dom.viewport.appendChild(renderer.domElement);

const stlLoader = new STLLoader();

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.copy(DEFAULT_VIEW_FOCUS);
controls.enableDamping = false;
controls.enableRotate = true;
controls.minDistance = 0.45;
controls.maxDistance = 8.0;

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
let lastFrameBox = null;
let pendingFrameRequest = 0;
let urdfJointControls = new Map();
let urdfLinkGroups = new Map();
let lastJointState = {};

init();

window.__g1dViewerDebug = {
  getCameraState: () => ({
    view: currentView,
    position: vectorToArray(camera.position),
    target: vectorToArray(controls.target),
    up: vectorToArray(camera.up),
    aspect: roundNumber(camera.aspect, 3),
  }),
  receivePose: (pose) => applyPose(pose, "YOLO 当前数据"),
  setView,
  frameScene,
};

async function init() {
  applyUrlParams();
  bindEvents();
  bindEmbeddedMessages();
  await loadRobot();
  if (NO_AUTO_FETCH_POSE) {
    setStatus("等待当前 YOLO 数据");
    try {
      await fetchRobotState({ silent: true });
    } catch {
      writeSceneState();
    }
    frameScene();
    notifyEmbeddedReady();
  } else {
    await refreshSceneData({ fallbackToSample: true, silent: true });
    notifyEmbeddedReady();
  }
  animate();
}

function applyUrlParams() {
  document.body.classList.toggle("compact-mode", IS_COMPACT_MODE);
  const sourceUrl = URL_PARAMS.get("source_url") || URL_PARAMS.get("xyz_url");
  if (sourceUrl) dom.sourceUrl.value = sourceUrl;
  const robotStateUrl = URL_PARAMS.get("robot_state_url");
  if (robotStateUrl) dom.robotStateUrl.value = robotStateUrl;
  const label = URL_PARAMS.get("label");
  if (label && [...dom.labelSelect.options].some((option) => option.value === label)) {
    dom.labelSelect.value = label;
  }
  const view = URL_PARAMS.get("view");
  if (["normal", "top", "side"].includes(view)) {
    currentView = view;
  }
}

function bindEvents() {
  dom.fetchBtn.addEventListener("click", () => refreshSceneData({ fallbackToSample: false, silent: false }));
  if (dom.baseCoordRefresh) {
    dom.baseCoordRefresh.addEventListener("click", () => refreshSceneData({ fallbackToSample: false, silent: false }));
  }
  if (dom.useOurPnp) {
    dom.useOurPnp.addEventListener("change", () => refreshSceneData({ fallbackToSample: false, silent: false }));
  }
  dom.normalViewBtn.addEventListener("click", () => setView("normal"));
  dom.topViewBtn.addEventListener("click", () => setView("top"));
  dom.sideViewBtn.addEventListener("click", () => setView("side"));
  for (const input of [dom.thicknessMm]) {
    input.addEventListener("input", () => {
      updateCameraMarker();
      if (currentPose) renderPose(currentPose, { frame: false });
    });
  }
  for (const toggle of [dom.showOldOld, dom.showOldNew, dom.showNewNew, dom.showNewOld, dom.showNewNewDist, dom.centerOnly]) {
    if (!toggle) continue;
    toggle.addEventListener("change", () => {
      if (currentPose) {
        renderPose(currentPose, { frame: false });
        updateMetrics(currentPose);
      }
    });
  }
  if (dom.autoRefresh) dom.autoRefresh.addEventListener("change", applyAutoRefresh);
  if (dom.autoInterval) dom.autoInterval.addEventListener("change", applyAutoRefresh);
  applyAutoRefresh();

  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(dom.viewport);
  resize();
}

function applyAutoRefresh() {
  if (autoRefreshTimer) {
    clearInterval(autoRefreshTimer);
    autoRefreshTimer = null;
  }
  if (!dom.autoRefresh || !dom.autoRefresh.checked) return;
  const intervalMs = Math.max(100, Number(dom.autoInterval?.value) || 500);
  autoRefreshTimer = setInterval(async () => {
    if (autoRefreshInFlight) return; // skip if the previous refresh is still running
    autoRefreshInFlight = true;
    try {
      // Only refresh the ARM/robot joint state (cheap DDS read). Do NOT re-run
      // YOLO and do NOT re-frame the camera, so the viewpoint stays put.
      await fetchRobotState({ silent: true });
    } catch {
      /* errors already surfaced via status text */
    } finally {
      autoRefreshInFlight = false;
    }
  }, intervalMs);
}

function bindEmbeddedMessages() {
  window.addEventListener("message", (event) => {
    const data = event.data;
    if (!data || typeof data !== "object") return;
    if (data.type !== "g1d-visualizer-pose") return;
    if (data.robotState && typeof data.robotState === "object") {
      applyRobotState(data.robotState);
    }
    if (data.pose && typeof data.pose === "object") {
      applyPose(data.pose, "YOLO 当前数据");
    }
  });
}

function notifyEmbeddedReady() {
  if (!IS_COMPACT_MODE || window.parent === window) return;
  window.parent.postMessage({ type: "g1d-visualizer-ready" }, "*");
}

async function refreshSceneData({ fallbackToSample = false, silent = false } = {}) {
  if (!silent) setStatus("读取数据");
  const [stateResult, poseResult] = await Promise.allSettled([
    fetchRobotState({ silent: true }),
    fetchPose({ fallbackToSample, silent: true }),
  ]);
  const failures = [stateResult, poseResult].filter((result) => result.status === "rejected");
  if (failures.length) {
    const message = failures.map((result) => result.reason?.message || String(result.reason)).join("；");
    if (!silent) setStatus(`读取失败：${message}`);
    return false;
  }
  if (!silent) setStatus("数据已更新");
  return true;
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
  applyPose(pose, "已加载默认数据");
}

async function fetchPose({ fallbackToSample = false, silent = false } = {}) {
  const url = new URL("/api/xyz", window.location.origin);
  url.searchParams.set("url", dom.sourceUrl.value.trim());
  if (dom.labelSelect.value) {
    url.searchParams.set("label", dom.labelSelect.value);
  }
  // Default uses the colleague's PnP; only opt into our PnP when checked.
  url.searchParams.set("our", dom.useOurPnp && dom.useOurPnp.checked ? "1" : "0");
  if (!silent) setStatus("读取 YOLO /xyz");
  try {
    const payload = await fetch(url).then((res) => res.json());
    if (!payload.ok) {
      throw new Error(payload.error || "读取失败");
    }
    applyPose(payload.pose, "YOLO 已更新");
    return true;
  } catch (error) {
    if (fallbackToSample) {
      await loadSample();
      setStatus(`YOLO 失败，已加载默认数据：${error.message}`);
      return true;
    }
    if (!silent) {
      setStatus(`YOLO 失败：${error.message}`);
    }
    throw error;
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
    return true;
  } catch (error) {
    if (!silent) setStatus(`状态失败：${error.message}`);
    throw error;
  }
}

function applyPose(pose, status, { frame = true } = {}) {
  currentPose = pose;
  renderPose(pose, { frame });
  updateMetrics(pose);
  setStatus(status);
}

const BOX_EDGE_OLD = 0xffa64d;     // ① 老内参 + 老外参(橙)
const BOX_EDGE_NEW = 0x4fd8ff;     // ② 老内参 + 新外参(青)
const BOX_EDGE_NEWINTR = 0x59e6a7; // ③ 新内参 + 新外参(绿)
const BOX_EDGE_NEWOLD = 0xffe14d;  // ④ 新内参 + 老外参(黄)
const BOX_EDGE_OURS = 0xb38bff;    // ⑤ 我们自己的 PnP(紫)
const BOX_EDGE_NEWDIST = 0xff6fd8; // ⑥ 新内参 + 新外参 + 畸变校正(粉)

// Build the full box orientation from our PnP axes (falls back to yaw only).
function ourPnpQuat(ourMethod, torsoQuat) {
  const axes = ourMethod.box_axes_base;
  if (axes && Array.isArray(axes.x) && Array.isArray(axes.y) && Array.isArray(axes.z)) {
    const m = new THREE.Matrix4().makeBasis(
      new THREE.Vector3(...axes.x),
      new THREE.Vector3(...axes.y),
      new THREE.Vector3(...axes.z),
    );
    return torsoQuat.clone().multiply(new THREE.Quaternion().setFromRotationMatrix(m));
  }
  return new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, Number(ourMethod.box_yaw_base_rad) || 0));
}

function renderPose(pose, { frame = false } = {}) {
  cigaretteGroup.clear();
  updateCameraMarker();

  // base_coords carries the YOLO-service PnP point under OLD vs NEW intrinsics,
  // each mapped to base via nominal URDF / our hand-eye (3 categories).
  // our_method is our OWN PnP (4th box, only when opted in).
  const bc = pose.base_coords && pose.base_coords.ok ? pose.base_coords : null;
  const ourMethod = pose.our_method && pose.our_method.ok ? pose.our_method : null;

  const wantOldOld = dom.showOldOld ? dom.showOldOld.checked : true; // ① 老内+老外
  const wantOldNew = dom.showOldNew ? dom.showOldNew.checked : true; // ② 老内+新外
  const wantNewNew = dom.showNewNew ? dom.showNewNew.checked : true; // ③ 新内+新外
  const wantNewOld = dom.showNewOld ? dom.showNewOld.checked : true; // ④ 新内+老外
  const wantOurs = dom.useOurPnp ? dom.useOurPnp.checked : false;    // ⑤ 我们的 PnP
  const wantNewDist = dom.showNewNewDist ? dom.showNewNewDist.checked : true; // ⑥ 新内+新外+畸变
  const centerOnly = dom.centerOnly ? dom.centerOnly.checked : false;

  const drawOldOld = wantOldOld && bc && Array.isArray(bc.c_old_old_m);
  const drawOldNew = wantOldNew && bc && Array.isArray(bc.c_old_new_m);
  const drawNewNew = wantNewNew && bc && Array.isArray(bc.c_new_new_m);
  const drawNewOld = wantNewOld && bc && Array.isArray(bc.c_new_old_m);
  const drawOurs = wantOurs && ourMethod && Array.isArray(ourMethod.center_base_m);
  const drawNewDist = wantNewDist && bc && Array.isArray(bc.c_new_new_dist_m);

  if (!drawOldOld && !drawOldNew && !drawNewNew && !drawNewOld && !drawOurs && !drawNewDist) {
    setStatus(bc || ourMethod ? "未勾选任何方法(或缺少对应数据)" : "缺少 base_coords / center_xyz_mm");
    if (dom.metricDelta) dom.metricDelta.textContent = "-";
    writeSceneState();
    return;
  }

  const dims = getBoxDimensions(pose);
  const torso = getLinkWorldTransform(DEFAULT_CAMERA_PARENT_LINK);
  const torsoPos = torso?.position?.clone() || new THREE.Vector3();
  const torsoQuat = torso?.quaternion || new THREE.Quaternion();
  const torsoToWorld = (m) => torsoPos.clone().add(new THREE.Vector3(...m).applyQuaternion(torsoQuat));
  // Colleague yaw (robot/torso frame) shared by the YOLO-PnP boxes.
  const yawQuat = torsoQuat.clone().multiply(
    new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, getBoxYawRobotRad(pose))),
  );

  // Draw one method: center marker always; box shape only when not center-only.
  const drawMethod = (centerM, quat, color, boxTag, markerTag, camOffsetM) => {
    const center = torsoToWorld(centerM);
    if (!centerOnly) drawCigaretteBox(center, quat, dims, color, boxTag);
    addPointMarker(cigaretteGroup, center, color, markerTag);
    if (Array.isArray(camOffsetM)) addCameraToBoxLine(torsoToWorld(camOffsetM), center, color);
    return center;
  };

  let cOldOld = null;
  let cOldNew = null;
  let cNewNew = null;
  let cNewOld = null;
  let cOurs = null;
  let cNewDist = null;
  let focus = null;

  // ① 老内参 + 老外参: OLD-intrinsics point + nominal URDF.
  if (drawOldOld) {
    cOldOld = drawMethod(bc.c_old_old_m, yawQuat, BOX_EDGE_OLD, "①老内+老外", "①", bc.cam_old_ex_m);
    focus = cOldOld;
  }
  // ② 老内参 + 新外参: OLD-intrinsics point + our hand-eye.
  if (drawOldNew) {
    cOldNew = drawMethod(bc.c_old_new_m, yawQuat, BOX_EDGE_NEW, "②老内+新外", "②", bc.cam_new_ex_m);
    if (!focus) focus = cOldNew;
  }
  // ③ 新内参 + 新外参: NEW-intrinsics point + our hand-eye.
  if (drawNewNew) {
    cNewNew = drawMethod(bc.c_new_new_m, yawQuat, BOX_EDGE_NEWINTR, "③新内+新外", "③", bc.cam_new_ex_m);
    if (!focus) focus = cNewNew;
  }
  // ④ 新内参 + 老外参: NEW-intrinsics point + nominal URDF.
  if (drawNewOld) {
    cNewOld = drawMethod(bc.c_new_old_m, yawQuat, BOX_EDGE_NEWOLD, "④新内+老外", "④", bc.cam_old_ex_m);
    if (!focus) focus = cNewOld;
  }
  // ⑤ 我们自己的 PnP + 我们手眼(勾选"使用我们的 PnP"才出现)。
  if (drawOurs) {
    cOurs = drawMethod(ourMethod.center_base_m, ourPnpQuat(ourMethod, torsoQuat), BOX_EDGE_OURS, "⑤我们PnP", "⑤", ourMethod.camera_offset_m);
    focus = cOurs; // prefer framing on our own result when shown
  }
  // ⑥ 新内参 + 新外参 + 畸变校正: NEW-intrinsics-undistorted point + our hand-eye.
  if (drawNewDist) {
    cNewDist = drawMethod(bc.c_new_new_dist_m, yawQuat, BOX_EDGE_NEWDIST, "⑥新内+新外+畸变", "⑥", bc.cam_new_ex_m);
    if (!focus) focus = cNewDist;
  }

  // metricDelta = 外参差(①→②, 同一 OLD 内参点, 仅外参不同)。
  if (dom.metricDelta) {
    dom.metricDelta.textContent = bc && bc.delta_ex_mm != null ? `${Math.round(bc.delta_ex_mm)} mm` : "-";
  }

  window.__g1dVisualizerState.cigaretteObjects = countObjects(cigaretteGroup);
  const primary = cNewNew || cNewDist || cOldNew || cNewOld || cOldOld || cOurs;
  if (primary) window.__g1dVisualizerState.lastCenterRobotM = [primary.x, primary.y, primary.z];
  writeSceneState();

  if (frame && focus) frameScene(focus);
}

function colorToHex(color) {
  return `#${color.toString(16).padStart(6, "0")}`;
}

function drawCigaretteBox(centerRobot, quat, dims, edgeColor, tagText) {
  const boxGroup = new THREE.Group();
  boxGroup.position.copy(centerRobot);
  boxGroup.quaternion.copy(quat);

  const body = new THREE.Mesh(new THREE.BoxGeometry(dims.long, dims.short, dims.thickness), materials.box);
  body.position.z = -dims.thickness / 2;
  boxGroup.add(body);

  const top = new THREE.Mesh(new THREE.BoxGeometry(dims.long * 1.002, dims.short * 1.002, 0.003), materials.boxTop);
  top.position.z = 0.002;
  boxGroup.add(top);

  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.BoxGeometry(dims.long, dims.short, dims.thickness)),
    new THREE.LineBasicMaterial({ color: edgeColor }),
  );
  edges.position.z = -dims.thickness / 2;
  boxGroup.add(edges);

  addLabel(boxGroup, tagText, new THREE.Vector3(0, 0, 0.075), colorToHex(edgeColor));
  cigaretteGroup.add(boxGroup);
}

function addCameraToBoxLine(cameraMount, centerRobot, color) {
  const geometry = new THREE.BufferGeometry().setFromPoints([cameraMount, centerRobot]);
  cigaretteGroup.add(new THREE.Line(geometry, new THREE.LineBasicMaterial({ color })));
  const distance = cameraMount.distanceTo(centerRobot);
  addLabel(
    cigaretteGroup,
    `${Math.round(distance * 1000)} mm`,
    centerRobot.clone().lerp(cameraMount, 0.5).add(new THREE.Vector3(0, 0, 0.05)),
    colorToHex(color),
  );
}

function formatBaseCoordMm(m) {
  if (!Array.isArray(m) || m.length < 3) return "-";
  return `[${(m[0] * 1000).toFixed(0)}, ${(m[1] * 1000).toFixed(0)}, ${(m[2] * 1000).toFixed(0)}] mm`;
}

function updateBaseCoordMetrics(pose) {
  const bc = pose && pose.base_coords && pose.base_coords.ok ? pose.base_coords : null;
  const om = pose && pose.our_method ? pose.our_method : null;
  const our = om && om.ok ? om : null;

  if (dom.baseCoordOldOld) dom.baseCoordOldOld.textContent = bc ? formatBaseCoordMm(bc.c_old_old_m) : "-";
  if (dom.baseCoordOldNew) dom.baseCoordOldNew.textContent = bc ? formatBaseCoordMm(bc.c_old_new_m) : "-";
  if (dom.baseCoordNewNew) dom.baseCoordNewNew.textContent = bc ? formatBaseCoordMm(bc.c_new_new_m) : "-";
  if (dom.baseCoordNewOld) dom.baseCoordNewOld.textContent = bc ? formatBaseCoordMm(bc.c_new_old_m) : "-";
  if (dom.baseCoordOurs) dom.baseCoordOurs.textContent = our ? formatBaseCoordMm(our.center_base_m) : "-";
  if (dom.baseCoordNewDist) dom.baseCoordNewDist.textContent = bc ? formatBaseCoordMm(bc.c_new_new_dist_m) : "-";
  if (dom.baseCoordDeltaEx) {
    dom.baseCoordDeltaEx.textContent = bc && bc.delta_ex_mm != null ? `${Math.round(bc.delta_ex_mm)} mm` : "-";
  }
  if (dom.baseCoordDeltaIn) {
    dom.baseCoordDeltaIn.textContent = bc && bc.delta_in_mm != null ? `${Math.round(bc.delta_in_mm)} mm` : "-";
  }
  if (dom.baseCoordDeltaDist) {
    dom.baseCoordDeltaDist.textContent = bc && bc.delta_dist_mm != null ? `${Math.round(bc.delta_dist_mm)} mm` : "-";
  }

  if (dom.baseCoordStatus) {
    if (bc) {
      dom.baseCoordStatus.textContent =
        `内参×外参对比 · torso_link 系 [X前, Y左, Z上] mm (外参差=①→②, 内参差=②→③, 畸变差=③→⑥)`;
    } else if (om && om.error) {
      dom.baseCoordStatus.textContent = `无坐标: ${om.error}`;
    } else {
      dom.baseCoordStatus.textContent = "无目标点(未检测到目标, 或服务端未加载标定矩阵)";
    }
  }
}

function updateMetrics(pose) {
  updateBaseCoordMetrics(pose);
  const bc = pose.base_coords && pose.base_coords.ok ? pose.base_coords : null;
  // Center metrics follow ③ 新内参+新外参 (the fully-calibrated estimate).
  if (bc && Array.isArray(bc.c_new_new_m)) {
    dom.metricLabel.textContent = getPoseLabel(pose);
    dom.metricForward.textContent = formatMm(bc.c_new_new_m[0] * 1000);   // torso +X 前
    dom.metricVertical.textContent = formatMm(-bc.c_new_new_m[2] * 1000); // torso -Z 下
    dom.metricNear.textContent = "-";
    dom.metricTurn.textContent = "-";
    dom.metricYaw.textContent = "-";
    dom.metricState.textContent = "③ 新内参+新外参";
    return;
  }
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
  lastJointState = stateJointValues;
  applyJointState(stateJointValues);
  updateColumnState();
  updateCameraMarker();
  if (currentPose) {
    renderPose(currentPose, { frame: false });
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
      scheduleFrameScene();
    },
    undefined,
    () => {
      addLinkProxy(group, linkName);
      window.__g1dVisualizerState.urdfFailedMeshes = (window.__g1dVisualizerState.urdfFailedMeshes || 0) + 1;
      window.__g1dVisualizerState.robotObjects = countObjects(robotGroup);
      writeSceneState();
      scheduleFrameScene();
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
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.032, 0.024, 0.024), materials.camera);
  body.position.copy(mount);
  body.quaternion.copy(frame.parentQuaternion);
  cameraMarkerGroup.add(body);
  addLabel(cameraMarkerGroup, "left camera", mount.clone().add(new THREE.Vector3(0.02, 0.02, 0.055)), "#9be2ff");
}

function getCameraMount() {
  return getCameraFrame(currentPose).origin;
}

function getCameraFrame(pose = null) {
  const parentTransform = getLinkWorldTransform(DEFAULT_CAMERA_PARENT_LINK);
  const parentPosition = parentTransform?.position || new THREE.Vector3();
  const parentQuaternion = parentTransform?.quaternion || new THREE.Quaternion();
  const localOffset = new THREE.Vector3(
    numberOrDefault(dom.cameraX.value, DEFAULT_CAMERA_OFFSET_M.x),
    numberOrDefault(dom.cameraY.value, DEFAULT_CAMERA_OFFSET_M.y),
    numberOrDefault(dom.cameraZ.value, DEFAULT_CAMERA_OFFSET_M.z),
  );
  const origin = parentPosition.clone().add(localOffset.clone().applyQuaternion(parentQuaternion));
  const cameraToVerticalDeg = Number(
    pose?.g1d_visualization?.camera?.camera_to_vertical_deg
      ?? pose?.robot_alignment?.camera_to_vertical_deg
      ?? DEFAULT_CAMERA_TO_VERTICAL_DEG,
  );
  const localAxes = cameraOpticalAxesInHeadLocal(cameraToVerticalDeg);
  return {
    origin,
    parentPosition,
    parentQuaternion,
    localOffset,
    cameraToVerticalDeg,
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
  addLabel(group, `cam Z ${Number(frame.cameraToVerticalDeg).toFixed(1)}°`, origin.clone().add(axes.zForward.clone().multiplyScalar(0.25)), "#72a3ff");
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

function scheduleFrameScene() {
  if (pendingFrameRequest) cancelAnimationFrame(pendingFrameRequest);
  pendingFrameRequest = requestAnimationFrame(() => {
    pendingFrameRequest = 0;
    frameScene();
  });
}

function frameScene() {
  lastFrameBox = computeFrameBox();
  setView(currentView, false);
}

function setView(mode, immediate = true) {
  currentView = mode;
  const frameBox = lastFrameBox || computeFrameBox();
  if (mode === "top") {
    controls.enableRotate = false;
    fitCameraToBox(frameBox, new THREE.Vector3(0.02, 0.0, 1.0), new THREE.Vector3(1, 0, 0), 1.06);
  } else if (mode === "side") {
    controls.enableRotate = true;
    fitCameraToBox(frameBox, new THREE.Vector3(0.02, -1.0, 0.08), new THREE.Vector3(0, 0, 1), 1.04);
  } else {
    controls.enableRotate = true;
    fitCameraToBox(frameBox, new THREE.Vector3(1.45, -1.25, 0.72), new THREE.Vector3(0, 0, 1), 1.02);
  }
  if (immediate) {
    setStatus(mode === "top" ? "俯视地面" : mode === "side" ? "侧视" : "正常地面视角");
  }
}

function computeFrameBox() {
  const box = new THREE.Box3();
  expandBoundsFromObject(box, robotGroup);
  expandBoundsFromObject(box, cameraMarkerGroup);
  expandBoundsFromObject(box, cigaretteGroup);
  if (box.isEmpty()) {
    box.setFromCenterAndSize(DEFAULT_VIEW_FOCUS, new THREE.Vector3(1.2, 0.9, 1.9));
  }
  const size = box.getSize(new THREE.Vector3());
  const pad = new THREE.Vector3(
    Math.max(0.04, size.x * 0.05),
    Math.max(0.04, size.y * 0.05),
    Math.max(0.04, size.z * 0.05),
  );
  box.min.sub(pad);
  box.max.add(pad);
  return box;
}

function expandBoundsFromObject(box, object) {
  object.updateMatrixWorld(true);
  object.traverse((child) => {
    if (!child.visible || child.isSprite || !child.geometry) return;
    if (!child.geometry.boundingBox) {
      child.geometry.computeBoundingBox();
    }
    if (!child.geometry.boundingBox) return;
    const childBox = child.geometry.boundingBox.clone().applyMatrix4(child.matrixWorld);
    if (!childBox.isEmpty()) {
      box.union(childBox);
    }
  });
}

function fitCameraToBox(box, direction, up, padding = 1.15) {
  const center = box.getCenter(new THREE.Vector3());
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  const safeDirection = normalizedVector(direction);
  const halfVerticalFov = THREE.MathUtils.degToRad(camera.fov * 0.5);
  const halfHorizontalFov = Math.atan(Math.tan(halfVerticalFov) * Math.max(0.1, camera.aspect));
  const radius = Math.max(0.35, sphere.radius * padding);
  const distance = Math.max(
    0.85,
    radius / Math.sin(Math.max(0.1, halfVerticalFov)),
    radius / Math.sin(Math.max(0.1, halfHorizontalFov)),
  );
  camera.up.copy(up);
  camera.position.copy(center).add(safeDirection.multiplyScalar(distance));
  controls.target.copy(center);
  camera.near = Math.max(0.005, distance - radius * 3.0);
  camera.far = Math.max(20, distance + radius * 4.0);
  camera.lookAt(center);
  camera.updateProjectionMatrix();
  controls.update();
}

function resize() {
  const rect = dom.viewport.getBoundingClientRect();
  const width = Math.max(1, rect.width);
  const height = Math.max(1, rect.height);
  renderer.setSize(width, height, true);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  if (lastFrameBox) {
    setView(currentView, false);
  }
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
    cameraParentLink: DEFAULT_CAMERA_PARENT_LINK,
    cameraOpticalAngleDeg: roundNumber(getCameraFrame(currentPose).cameraToVerticalDeg, 3),
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

// ---------------------------------------------------------------------------
// Planned-trajectory playback (added for the suction-pick pipeline).
// Loads a trajectory JSON exported by `pick.trajectory` and animates the arm
// joints over time on top of the existing 3D robot, plus draws the planned box
// and the pregrasp/grasp/lift targets. Purely additive to the live viewer.
// ---------------------------------------------------------------------------
const planGroup = new THREE.Group();
root.add(planGroup);

const pbDom = {
  panel: document.querySelector("#playbackPanel"),
  loadServer: document.querySelector("#pbLoadServer"),
  file: document.querySelector("#pbFile"),
  play: document.querySelector("#pbPlay"),
  restart: document.querySelector("#pbRestart"),
  speed: document.querySelector("#pbSpeed"),
  slider: document.querySelector("#pbSlider"),
  time: document.querySelector("#pbTime"),
  phase: document.querySelector("#pbPhase"),
  suction: document.querySelector("#pbSuction"),
};

const pb = {
  traj: null,
  jointNames: [],
  samples: [],
  times: [],
  events: [],
  duration: 0,
  playing: false,
  t: 0,
  speed: 1,
  lastWall: 0,
  raf: 0,
};

if (pbDom.panel) bindPlaybackEvents();

function bindPlaybackEvents() {
  pbDom.loadServer.addEventListener("click", () => loadTrajectoryFromUrl("./pick_trajectory.json"));
  pbDom.file.addEventListener("change", onPlaybackFile);
  pbDom.play.addEventListener("click", togglePlay);
  pbDom.restart.addEventListener("click", () => seekPlayback(0, { pause: true }));
  pbDom.speed.addEventListener("change", () => { pb.speed = Number(pbDom.speed.value) || 1; });
  pbDom.slider.addEventListener("input", () => {
    seekPlayback(Number(pbDom.slider.value) * pb.duration, { pause: true });
  });
}

async function loadTrajectoryFromUrl(url) {
  try {
    setStatus(`读取轨迹 ${url}`);
    const data = await fetch(url, { cache: "no-store" }).then((res) => {
      if (!res.ok) throw new Error(`${res.status}`);
      return res.json();
    });
    setTrajectory(data);
    setStatus(`轨迹已载入(${pb.duration.toFixed(2)}s, ${pb.samples.length} 帧)`);
  } catch (error) {
    setStatus(`轨迹载入失败：${error.message}(确认 run_pick 已导出 pick_trajectory.json)`);
  }
}

function onPlaybackFile(event) {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    try {
      setTrajectory(JSON.parse(String(reader.result)));
      setStatus(`轨迹已载入：${file.name}(${pb.duration.toFixed(2)}s)`);
    } catch (error) {
      setStatus(`轨迹解析失败：${error.message}`);
    }
  };
  reader.readAsText(file);
}

function setTrajectory(traj) {
  if (!traj || !Array.isArray(traj.samples) || !Array.isArray(traj.joint_names)) {
    throw new Error("轨迹格式不正确(缺少 samples / joint_names)");
  }
  stopPlayback();
  // Stop live auto-refresh so it doesn't fight the playback.
  if (dom.autoRefresh && dom.autoRefresh.checked) {
    dom.autoRefresh.checked = false;
    applyAutoRefresh();
  }
  pb.traj = traj;
  pb.jointNames = traj.joint_names.slice();
  pb.samples = traj.samples.map((s) => ({ t: Number(s.t) || 0, phase: s.phase || "", q: s.q }));
  pb.samples.sort((a, b) => a.t - b.t);
  pb.times = pb.samples.map((s) => s.t);
  pb.events = Array.isArray(traj.events) ? traj.events.slice() : [];
  pb.duration = Number(traj.duration_s) || (pb.times.length ? pb.times[pb.times.length - 1] : 0);
  pb.t = 0;

  pbDom.play.disabled = false;
  pbDom.restart.disabled = false;
  pbDom.slider.disabled = false;
  pbDom.play.textContent = "播放";

  drawPlannedScene(traj.meta || {});
  renderPlaybackFrame(0);
}

function torsoToWorld() {
  const torso = getLinkWorldTransform("torso_link");
  const pos = torso?.position?.clone() || new THREE.Vector3();
  const quat = torso?.quaternion || new THREE.Quaternion();
  return {
    point: (m) => pos.clone().add(new THREE.Vector3(m[0], m[1], m[2]).applyQuaternion(quat)),
    dir: (v) => new THREE.Vector3(v[0], v[1], v[2]).applyQuaternion(quat).normalize(),
    quat,
  };
}

function drawPlannedScene(meta) {
  planGroup.clear();
  if (!meta || !Array.isArray(meta.box_center_base)) return;
  const tf = torsoToWorld();

  const dims = Array.isArray(meta.box_dims_m) ? meta.box_dims_m : [0.16, 0.09];
  const longSide = Math.max(dims[0], dims[1]);
  const shortSide = Math.min(dims[0], dims[1]);
  const thickness = 0.02;

  // Box pose: x = long axis, z = top normal, y = z x x (in torso frame).
  const center = tf.point(meta.box_center_base);
  let quat = new THREE.Quaternion();
  if (Array.isArray(meta.box_long_axis_base) && Array.isArray(meta.box_normal_base)) {
    const x = new THREE.Vector3(...meta.box_long_axis_base).normalize();
    const z = new THREE.Vector3(...meta.box_normal_base).normalize();
    const y = new THREE.Vector3().crossVectors(z, x).normalize();
    z.crossVectors(x, y).normalize();
    const m = new THREE.Matrix4().makeBasis(x, y, z);
    quat = tf.quat.clone().multiply(new THREE.Quaternion().setFromRotationMatrix(m));
  }
  const boxGroup = new THREE.Group();
  boxGroup.position.copy(center);
  boxGroup.quaternion.copy(quat);
  const body = new THREE.Mesh(
    new THREE.BoxGeometry(longSide, shortSide, thickness),
    new THREE.MeshStandardMaterial({ color: 0xd8aa39, roughness: 0.58, transparent: true, opacity: 0.55 }),
  );
  body.position.z = -thickness / 2;
  boxGroup.add(body);
  boxGroup.add(new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.BoxGeometry(longSide, shortSide, thickness)),
    new THREE.LineBasicMaterial({ color: 0x4fd8ff }),
  ).translateZ(-thickness / 2));
  planGroup.add(boxGroup);

  const mark = (arr, color, label) => {
    if (!Array.isArray(arr)) return;
    addPointMarker(planGroup, tf.point(arr), color, label);
  };
  mark(meta.pregrasp_pos, 0x4fd8ff, "pregrasp");
  mark(meta.grasp_pos, 0x35d477, "grasp");
  mark(meta.lift_pos, 0xffa64d, "lift");

  if (Array.isArray(meta.target_axis_base) && Array.isArray(meta.grasp_pos)) {
    const origin = tf.point(meta.grasp_pos).add(tf.dir(meta.target_axis_base).multiplyScalar(-0.12));
    planGroup.add(new THREE.ArrowHelper(tf.dir(meta.target_axis_base), origin, 0.12, 0x35d477, 0.03, 0.02));
  }
}

function indexAtTime(t) {
  const times = pb.times;
  if (!times.length) return -1;
  let lo = 0;
  let hi = times.length - 1;
  if (t <= times[0]) return 0;
  if (t >= times[hi]) return hi;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (times[mid] <= t) lo = mid; else hi = mid - 1;
  }
  return lo;
}

function renderPlaybackFrame(t) {
  if (!pb.samples.length) return;
  const idx = indexAtTime(t);
  const sample = pb.samples[idx];
  const armJoints = {};
  for (let i = 0; i < pb.jointNames.length; i += 1) {
    armJoints[pb.jointNames[i]] = Number(sample.q[i]);
  }
  applyJointState({ ...lastJointState, ...armJoints });

  pbDom.slider.value = pb.duration > 0 ? String(t / pb.duration) : "0";
  pbDom.time.textContent = `${t.toFixed(2)} / ${pb.duration.toFixed(2)} s`;
  pbDom.phase.textContent = sample.phase || "-";

  let suctionOn = false;
  for (const ev of pb.events) {
    if (Number(ev.t) <= t) {
      if (ev.type === "suction_on") suctionOn = true;
      else if (ev.type === "suction_off") suctionOn = false;
    }
  }
  pbDom.suction.textContent = suctionOn ? "吸 (ON)" : "松 (OFF)";
  pbDom.suction.style.color = suctionOn ? "#35d477" : "#9fb8c8";
}

function togglePlay() {
  if (!pb.samples.length) return;
  if (pb.playing) {
    stopPlayback();
  } else {
    if (pb.t >= pb.duration - 1e-6) pb.t = 0;
    pb.playing = true;
    pb.lastWall = performance.now();
    pbDom.play.textContent = "暂停";
    pb.raf = requestAnimationFrame(playbackTick);
  }
}

function stopPlayback() {
  pb.playing = false;
  if (pb.raf) cancelAnimationFrame(pb.raf);
  pb.raf = 0;
  if (pbDom.play) pbDom.play.textContent = "播放";
}

function seekPlayback(t, { pause = false } = {}) {
  pb.t = Math.max(0, Math.min(pb.duration, t));
  if (pause) stopPlayback();
  renderPlaybackFrame(pb.t);
}

function playbackTick(now) {
  if (!pb.playing) return;
  const dt = Math.max(0, (now - pb.lastWall) / 1000) * pb.speed;
  pb.lastWall = now;
  pb.t += dt;
  if (pb.t >= pb.duration) {
    pb.t = pb.duration;
    renderPlaybackFrame(pb.t);
    stopPlayback();
    return;
  }
  renderPlaybackFrame(pb.t);
  pb.raf = requestAnimationFrame(playbackTick);
}

// ---------------------------------------------------------------------------
// Front-end pick control: 识别烟盒 / 模拟执行 / 真机执行.
// These call the visualizer server's POST /api/plan and /api/execute, which run
// our pick pipeline (perceive -> plan -> export | execute).
// ---------------------------------------------------------------------------
const ARM_JOINT_ORDER = [
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
];

const pickDom = {
  panel: document.querySelector("#pickPanel"),
  detect: document.querySelector("#pkDetect"),
  sim: document.querySelector("#pkSim"),
  run: document.querySelector("#pkRun"),
  status: document.querySelector("#pickStatus"),
};

if (pickDom.panel) {
  pickDom.detect.addEventListener("click", detectBox);
  pickDom.sim.addEventListener("click", simulateExecute);
  pickDom.run.addEventListener("click", realExecute);
}

function setPickStatus(text) {
  if (pickDom.status) pickDom.status.textContent = text;
}

function setPickBusy(busy) {
  for (const btn of [pickDom.detect, pickDom.sim, pickDom.run]) {
    if (btn) btn.disabled = busy;
  }
}

function getCurrentArmJoints() {
  const out = [];
  for (const name of ARM_JOINT_ORDER) {
    const value = lastJointState[name];
    if (!Number.isFinite(value)) return null;
    out.push(Number(value));
  }
  return out;
}

function pickPayload() {
  const payload = {};
  if (dom.labelSelect.value) payload.label = dom.labelSelect.value;
  const q = getCurrentArmJoints();
  if (q) payload.q_current = q;
  return payload;
}

async function postPick(path, payload) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json().catch(() => ({ ok: false, error: `HTTP ${res.status}` }));
  if (!data.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

async function detectBox() {
  setPickBusy(true);
  setPickStatus("识别中…");
  try {
    const ok = await refreshSceneData({ fallbackToSample: false, silent: true });
    setPickStatus(ok ? "识别完成,已显示烟盒" : "识别失败:检查 YOLO(18081)");
  } catch (error) {
    setPickStatus(`识别失败：${error.message}`);
  } finally {
    setPickBusy(false);
  }
}

async function simulateExecute() {
  setPickBusy(true);
  setPickStatus("规划中(感知 + IK)…");
  try {
    const data = await postPick("/api/plan", pickPayload());
    setTrajectory(data.trajectory);
    setPickStatus(`规划完成(种子:${data.seed_source}),开始回放`);
    if (!pb.playing) togglePlay();
  } catch (error) {
    setPickStatus(`模拟失败：${error.message}`);
  } finally {
    setPickBusy(false);
  }
}

async function realExecute() {
  const confirmed = window.confirm(
    "真机执行:机械臂将真实运动并启动吸盘。\n请确认周围安全、急停可达。是否继续?",
  );
  if (!confirmed) {
    setPickStatus("已取消真机执行");
    return;
  }
  setPickBusy(true);
  setPickStatus("真机执行中…运动期间请勿操作(完成后才会返回)");
  try {
    const data = await postPick("/api/execute", pickPayload());
    if (data.trajectory) setTrajectory(data.trajectory);
    setPickStatus("真机执行完成");
  } catch (error) {
    setPickStatus(`真机执行失败：${error.message}`);
  } finally {
    setPickBusy(false);
  }
}

// ---------------------------------------------------------------------------
// Ready pose: capture the current arm configuration as a table-safe waypoint.
// Operator lifts the arm to a good standoff over the table, a helper clicks 保存,
// and the server persists the 7 right-arm joints to pick/ready_pose.json.
// ---------------------------------------------------------------------------
const readyDom = {
  panel: document.querySelector("#readyPanel"),
  save: document.querySelector("#rpSave"),
  load: document.querySelector("#rpLoad"),
  status: document.querySelector("#readyStatus"),
};

if (readyDom.panel) {
  readyDom.save.addEventListener("click", saveReadyPose);
  readyDom.load.addEventListener("click", loadReadyPose);
}

function setReadyStatus(text) {
  if (readyDom.status) readyDom.status.textContent = text;
}

function formatJoints(q) {
  if (!Array.isArray(q)) return "-";
  return `[${q.map((v) => Number(v).toFixed(3)).join(", ")}]`;
}

async function saveReadyPose() {
  if (readyDom.save) readyDom.save.disabled = true;
  setReadyStatus("保存中…");
  try {
    const payload = { all_joints: { ...lastJointState } };
    const q = getCurrentArmJoints();
    if (q) payload.q_current = q;
    const col = Number(dom.columnExtensionMm?.value);
    if (Number.isFinite(col)) payload.column_extension_mm = col;
    const data = await postPick("/api/save_ready_pose", payload);
    setReadyStatus(`已保存预备位姿(来源:${data.source}) q=${formatJoints(data.q)}`);
  } catch (error) {
    setReadyStatus(`保存失败：${error.message}`);
  } finally {
    if (readyDom.save) readyDom.save.disabled = false;
  }
}

async function loadReadyPose() {
  setReadyStatus("读取已存预备位姿…");
  try {
    const res = await fetch("/api/ready_pose", { cache: "no-store" });
    const data = await res.json().catch(() => ({ ok: false, error: `HTTP ${res.status}` }));
    if (!data.ok) throw new Error(data.error || `HTTP ${res.status}`);
    setReadyStatus(`已存预备位姿(${data.saved_at}) q=${formatJoints(data.q)}`);
  } catch (error) {
    setReadyStatus(`读取失败：${error.message}`);
  }
}
