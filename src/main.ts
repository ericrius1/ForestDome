import * as THREE from 'three/webgpu';
import { PointerLockControls } from 'three/addons/controls/PointerLockControls.js';
import { VRButton } from 'three/addons/webxr/VRButton.js';
import { createTerrain, terrainHeight } from './terrain';
import { createForest } from './trees';
import { createGrass } from './grass';
import { createFlowers } from './flowers';
import { createNBodyDome } from './nbodyDome';
import { createLanterns } from './lanterns';
import { createSculpture } from './sculpture';
import { createPlayerPhysics } from './player';
import { createCursor } from './cursor';
import { createDayNight } from './sky';
import { Inspector } from 'three/addons/inspector/Inspector.js';

const renderer = new THREE.WebGPURenderer({ antialias: true });
await renderer.init();
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.2;
renderer.xr.enabled = true;
document.body.appendChild(renderer.domElement);
document.body.appendChild(VRButton.createButton(renderer));

const scene = new THREE.Scene();

const EYE = 1.7;
const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 400);
const camX = 7, camZ = 23.5;
camera.position.set(camX, terrainHeight(camX, camZ) + EYE, camZ);
camera.lookAt(-2, terrainHeight(-2, -4) + 3.2, -4);

// XR rig: in VR the headset drives the camera relative to this group, and
// thumbstick locomotion moves the group. On desktop it stays at the origin.
const rig = new THREE.Group();
rig.add(camera);
scene.add(rig);

const sun = new THREE.DirectionalLight(0xffe9c4, 3.8);
sun.position.set(-34, 26, -32);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.left = -55;
sun.shadow.camera.right = 55;
sun.shadow.camera.top = 55;
sun.shadow.camera.bottom = -55;
sun.shadow.camera.near = 1;
sun.shadow.camera.far = 140;
sun.shadow.bias = -0.0004;
sun.shadow.normalBias = 0.05;
scene.add(sun);

const hemi = new THREE.HemisphereLight(0xd2ddd8, 0x4a5230, 0.85);
scene.add(hemi);

const sky = createDayNight(scene, sun, hemi);

scene.add(createTerrain());
scene.add(createForest());
scene.add(createGrass());
scene.add(createFlowers());

const cursor = createCursor();
const dome = createNBodyDome(renderer, scene);
sky.attachPane(dome.pane);

// The Helix Spire ahead of spawn, with the magic lanterns hugging its flanks.
const lookDir = new THREE.Vector3(-2 - camX, 0, -4 - camZ).normalize();
const lanternRight = new THREE.Vector3(-lookDir.z, 0, lookDir.x);
const sculptCenter = new THREE.Vector3(camX, 0, camZ).addScaledVector(lookDir, 22);
const entryAngle = Math.atan2(camZ - sculptCenter.z, camX - sculptCenter.x);
const sculpture = createSculpture(sculptCenter, entryAngle);
scene.add(sculpture.group);

const lanterns = createLanterns(scene, [
  sculptCenter.clone().addScaledVector(lanternRight, -11).addScaledVector(lookDir, -3.5),
  sculptCenter.clone().addScaledVector(lanternRight, 11).addScaledVector(lookDir, -3.5),
]);
lanterns.attachPane(dome.pane);

// Capsule character physics against the sculpture's merged BVH (terrain is
// handled analytically inside).
const physics = createPlayerPhysics(sculpture.collider);

// ---------------- VR controllers ----------------

interface ControllerState {
  controller: THREE.Group;
  line: THREE.Line;
  sign: number; // +1 attract, -1 repulse, 0 idle
  handedness: XRHandedness;
}

const ATTRACT_COLOR = new THREE.Color('#4fd8ff');
const REPULSE_COLOR = new THREE.Color('#ffa14f');
const IDLE_COLOR = new THREE.Color(0xffffff);

function makeController(index: number): ControllerState {
  const controller = renderer.xr.getController(index);
  rig.add(controller);
  const geo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, -1),
  ]);
  const mat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.6 });
  const line = new THREE.Line(geo, mat);
  line.scale.z = 8;
  controller.add(line);
  const state: ControllerState = { controller, line, sign: 0, handedness: 'none' };
  // Role is fixed per hand: right attracts, left repulses. Trigger or grip
  // activates it, so both hands can run at once or solo.
  const roleSign = () => (state.handedness === 'left' ? -1 : 1);
  let pressed = 0;
  controller.addEventListener('connected', (e) => {
    state.handedness = (e as THREE.Event & { data?: XRInputSource }).data?.handedness ?? 'none';
  });
  controller.addEventListener('disconnected', () => { pressed = 0; state.sign = 0; });
  const press = () => { pressed++; state.sign = roleSign(); };
  const release = () => { pressed = Math.max(0, pressed - 1); if (pressed === 0) state.sign = 0; };
  controller.addEventListener('selectstart', press);
  controller.addEventListener('selectend', release);
  controller.addEventListener('squeezestart', press);
  controller.addEventListener('squeezeend', release);
  return state;
}

const controllers = [makeController(0), makeController(1)];

renderer.xr.addEventListener('sessionstart', () => {
  rig.position.set(camera.position.x, terrainHeight(camera.position.x, camera.position.z), camera.position.z);
  camera.position.set(0, 0, 0);
  overlay.style.display = 'none';
});

renderer.xr.addEventListener('sessionend', () => {
  camera.position.set(rig.position.x, terrainHeight(rig.position.x, rig.position.z) + EYE, rig.position.z);
  camera.rotation.set(0, 0, 0);
  rig.position.set(0, 0, 0);
  rig.rotation.set(0, 0, 0);
  if (!fallbackLook) overlay.style.display = '';
});

// ---------------- inspector + tweakpane overlay toggle ----------------

// The official three.js inspector (performance, memory, timeline) docked at
// the bottom of the page. Always available — dev and production alike — and
// hidden together with the rest of the HUD via the "/" toggle. The renderer
// only auto-attaches the inspector DOM when it's set before init(), so
// attach it manually here.
const devInspector = new Inspector();
renderer.inspector = devInspector;
document.body.appendChild(devInspector.domElement);
// Start minimized (collapsed pill); click it to expand. Collapse even if a
// previous session left the panel open in the persisted layout.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const profiler = (devInspector as any).profiler;
if (profiler.panel.classList.contains('visible')) profiler.togglePanel();

// Bottom-left key/control reference, shown alongside the rest of the HUD.
const infoBox = document.createElement('div');
infoBox.style.cssText = `
  position:fixed; left:12px; bottom:12px; z-index:25; padding:10px 14px;
  background:rgba(14,20,16,0.72); color:#dfe8df; border-radius:8px;
  font:400 12px/1.7 system-ui, sans-serif; letter-spacing:0.02em;
  pointer-events:none; white-space:nowrap;`;

function controlRows(rows: [string, string][]): string {
  return rows.map(([k, d]) =>
    `<tr><td style="padding-right:12px;opacity:0.7;text-align:right">${k}</td><td>${d}</td></tr>`,
  ).join('');
}

infoBox.innerHTML = `
  <table style="border-spacing:0"><tbody>
    ${controlRows([
      ['W A S D', 'move'],
      ['Space', 'jump'],
      ['Shift', 'run'],
      ['mouse', 'look'],
      ['hold click', 'apply force'],
      ['T', 'attract / repulse'],
      ['R', 'reseed particles'],
      ['P', 'pause sim'],
      ['/', 'toggle HUD'],
      ['Esc', 'release mouse'],
    ])}
  </tbody></table>`;
document.body.appendChild(infoBox);

if (navigator.xr) {
  navigator.xr.isSessionSupported('immersive-vr').then((supported) => {
    if (!supported) return;
    infoBox.insertAdjacentHTML('beforeend', `
      <div style="margin-top:8px;padding-top:6px;border-top:1px solid rgba(255,255,255,0.15);opacity:0.9;font-weight:600">VR</div>
      <table style="border-spacing:0"><tbody>
        ${controlRows([
          ['right trigger/grip', 'attract'],
          ['left trigger/grip', 'repulse'],
          ['left stick', 'move'],
          ['right stick', 'turn'],
        ])}
      </tbody></table>`);
  }).catch(() => {});
}

let overlaysVisible = true;

function setOverlaysVisible(visible: boolean) {
  overlaysVisible = visible;
  dome.pane.element.parentElement!.style.display = visible ? '' : 'none';
  infoBox.style.display = visible ? '' : 'none';
  devInspector.domElement.style.display = visible ? '' : 'none';
}

window.addEventListener('keydown', (e) => {
  if (e.code === 'Slash' && !e.repeat) {
    setOverlaysVisible(!overlaysVisible);
  }
});

// ---------------- FPS controls ----------------

const controls = new PointerLockControls(camera, renderer.domElement);

const overlay = document.createElement('div');
overlay.style.cssText = `
  position:fixed; inset:0; display:flex; align-items:center; justify-content:center;
  background:rgba(20,28,24,0.25); color:#eef3ee; cursor:pointer; user-select:none;
  font:500 18px/1.6 system-ui, sans-serif; text-align:center; letter-spacing:0.02em;`;
overlay.innerHTML = '<div>Click to walk<br><span style="font-size:14px;opacity:0.8">WASD move &middot; Space jump &middot; Shift run &middot; mouse look &middot; hold click for force &middot; t attract/repulse &middot; Esc release &middot; / toggle HUD<br>VR: right hand attracts &middot; left hand repulses &middot; left stick move &middot; right stick turn</span></div>';
document.body.appendChild(overlay);

let fallbackLook = false;
let dragLooking = false;
let everLocked = false;

document.addEventListener('pointerlockerror', () => {
  // Lock worked before, so this is the browser's ~1.3s cooldown after an
  // Esc exit — not missing support. The next click will succeed.
  if (everLocked) return;
  if (fallbackLook) return;
  fallbackLook = true;
  overlay.style.display = 'none';
  const euler = new THREE.Euler(0, 0, 0, 'YXZ');
  renderer.domElement.addEventListener('mousedown', () => { dragLooking = true; });
  window.addEventListener('mouseup', () => { dragLooking = false; });
  window.addEventListener('mousemove', (e) => {
    if (!dragLooking) return;
    euler.setFromQuaternion(camera.quaternion);
    euler.y -= e.movementX * 0.0025;
    euler.x = THREE.MathUtils.clamp(euler.x - e.movementY * 0.0025, -1.4, 1.4);
    camera.quaternion.setFromEuler(euler);
  });
});

overlay.addEventListener('click', () => { if (!fallbackLook) controls.lock(); });
controls.addEventListener('lock', () => { everLocked = true; overlay.style.display = 'none'; });
renderer.domElement.addEventListener('click', () => {
  if (!fallbackLook && !controls.isLocked && !renderer.xr.isPresenting) controls.lock();
});

const keys = new Set<string>();
window.addEventListener('keydown', (e) => {
  if (e.code === 'Space') e.preventDefault();
  keys.add(e.code);
});
window.addEventListener('keyup', (e) => keys.delete(e.code));

let pointerDown = false;
window.addEventListener('mousedown', (e) => { if (e.button === 0) pointerDown = true; });
window.addEventListener('mouseup', () => { pointerDown = false; });

let velF = 0, velR = 0;

// vertical physics state (desktop + VR share the gravity constants)
const GRAVITY = 24;
const JUMP_VEL = 7.5;
let velY = 0;
let grounded = true;
const feet = new THREE.Vector3();
let vrVelY = 0;

const clock = new THREE.Clock();

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ---------------- forcer input (mouse + VR controllers) ----------------

const rayOrigin = new THREE.Vector3();
const rayDir = new THREE.Vector3();
const tmpQuat = new THREE.Quaternion();

function updateForcers() {
  dome.clearForcers();

  if (renderer.xr.isPresenting) {
    cursor.set(null);
    for (let i = 0; i < controllers.length; i++) {
      const { controller, line, sign } = controllers[i];
      controller.getWorldPosition(rayOrigin);
      controller.getWorldQuaternion(tmpQuat);
      rayDir.set(0, 0, -1).applyQuaternion(tmpQuat);
      const hit = dome.intersectRay(rayOrigin, rayDir);
      const mat = line.material as THREE.LineBasicMaterial;
      if (hit) {
        line.scale.z = dome.simToWorld(hit, rayDir).sub(rayOrigin).length();
        mat.color.copy(sign > 0 ? ATTRACT_COLOR : sign < 0 ? REPULSE_COLOR : IDLE_COLOR);
        mat.opacity = sign !== 0 ? 1.0 : 0.6;
        if (sign !== 0) dome.setForcer(i + 1, hit, sign);
      } else {
        line.scale.z = 8;
        mat.color.copy(IDLE_COLOR);
        mat.opacity = 0.3;
      }
    }
    return;
  }

  camera.getWorldPosition(rayOrigin);
  camera.getWorldDirection(rayDir);
  const hit = dome.intersectRay(rayOrigin, rayDir);
  if (!hit) {
    cursor.set(null);
    return;
  }
  cursor.set(dome.params.pointerMode, pointerDown);
  if (pointerDown) {
    dome.setForcer(0, hit, dome.params.pointerMode === 'attract' ? 1 : -1);
  }
}

// ---------------- VR locomotion ----------------

const headYaw = new THREE.Vector3();

function updateLocomotion(dt: number) {
  const session = renderer.xr.getSession();
  if (!session) return;
  for (const source of session.inputSources) {
    const axes = source.gamepad?.axes;
    if (!axes || axes.length < 4) continue;
    const ax = axes[2], ay = axes[3];
    if (Math.abs(ax) < 0.1 && Math.abs(ay) < 0.1) continue;
    if (source.handedness === 'right') {
      rig.rotation.y -= ax * dt * 2.0;
    } else {
      camera.getWorldDirection(headYaw);
      headYaw.y = 0;
      headYaw.normalize();
      const speed = 3.5;
      rig.position.addScaledVector(headYaw, -ay * speed * dt);
      headYaw.cross(new THREE.Vector3(0, 1, 0));
      rig.position.addScaledVector(headYaw, -ax * speed * dt);
    }
  }
  vrVelY = Math.max(vrVelY - GRAVITY * dt, -28);
  rig.position.y += vrVelY * dt;
  vrVelY = physics.resolve(rig.position, vrVelY).velY;
}

(window as unknown as { __debug: object }).__debug = {
  camera, rig, scene, sky, controls, keys, dome, lanterns, sculpture, physics, renderer,
  isFallback: () => fallbackLook,
};

renderer.setAnimationLoop(() => {
  const dt = Math.min(clock.getDelta(), 0.05);

  if (renderer.xr.isPresenting) {
    updateLocomotion(dt);
  } else {
    if (controls.isLocked || fallbackLook) {
      const speed = keys.has('ShiftLeft') || keys.has('ShiftRight') ? 10 : 4.5;
      const tF = (keys.has('KeyW') ? 1 : 0) - (keys.has('KeyS') ? 1 : 0);
      const tR = (keys.has('KeyD') ? 1 : 0) - (keys.has('KeyA') ? 1 : 0);
      const k = Math.min(1, dt * 9);
      velF += (tF * speed - velF) * k;
      velR += (tR * speed - velR) * k;
      controls.moveForward(velF * dt);
      controls.moveRight(velR * dt);
      if (keys.has('Space') && grounded) {
        velY = JUMP_VEL;
        grounded = false;
      }
    }
    velY = Math.max(velY - GRAVITY * dt, -28);
    feet.set(camera.position.x, camera.position.y - EYE + velY * dt, camera.position.z);
    const res = physics.resolve(feet, velY);
    velY = res.velY;
    grounded = res.grounded;
    camera.position.set(feet.x, feet.y + EYE, feet.z);
  }

  updateForcers();
  dome.update(dt);
  lanterns.update(dt);
  sculpture.update(dt);
  sky.update(dt);

  renderer.render(scene, camera);
});
