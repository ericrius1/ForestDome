import * as THREE from 'three/webgpu';
import {
  Fn, uniform, instancedArray, instanceIndex,
  vec4, mix, smoothstep, uv,
} from 'three/tsl';
import { Pane } from 'tweakpane';
import { terrainHeight } from './terrain';
import { OctreeDomeSolver } from './octreeSolver';

// World-space footprint of the dome. It envelops most of the forest; the
// shell only nears the ground beyond the tree line, so trees live inside it.
export const DOME = {
  x: -10, z: -1, r: 84,
};

// Open meadow kept free of trees at the dome's center (the old footprint).
export const CLEARING_R = 22.5;

// The simulation runs in the blog post's units (shell radius ~0.9) and is
// scaled up into world space for rendering and interaction.
const SIM_R = 0.9;
const SCALE = DOME.r / SIM_R;

// Forcer slots: 0 = mouse/gaze, 1 = left controller, 2 = right controller.
export const FORCER_SLOTS = 3;

export interface NBodyDomeParams {
  count: number;
  steps: number;
  paused: boolean;
  totalMass: number;
  gravity: number;
  dt: number;
  softening: number;
  theta: number;
  damping: number;
  maxSpeed: number;
  shellK: number;
  shellR: number;
  floorK: number;
  spin: number;
  dispersion: number;
  flow: number;
  swirl: number;
  massMin: number;
  massMax: number;
  pointerMode: 'attract' | 'repulse';
  strength: number;
  pointerSoftening: number;
  pointerMassScale: number;
  sizeScale: number;
  minSize: number;
  colorScale: number;
  colorLow: string;
  colorHigh: string;
  brightness: number;
}

export interface NBodyDome {
  update: (dt: number) => void;
  reseed: () => void;
  /** Reset all forcer slots; call once per frame before setForcer. */
  clearForcers: () => void;
  /** Apply attract (+1) / repulse (-1) at a sim-space point this frame. */
  setForcer: (slot: number, simPoint: THREE.Vector3, sign: number) => void;
  /** World-space ray vs dome shell. Returns sim-space hit point, or null. */
  intersectRay: (origin: THREE.Vector3, dir: THREE.Vector3) => THREE.Vector3 | null;
  /** Sim-space point -> world space (for cursors / ray endpoints). */
  simToWorld: (simPoint: THREE.Vector3, out?: THREE.Vector3) => THREE.Vector3;
  center: THREE.Vector3;
  params: NBodyDomeParams;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  pane: any;
}

function randn(): number {
  const u = Math.random() || 1e-9;
  const v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function createNBodyDome(renderer: THREE.WebGPURenderer, scene: THREE.Scene): NBodyDome {
  const params: NBodyDomeParams = {
    // Barnes-Hut octree solver (ported from the blog post) makes the gravity
    // cost ~N·log N instead of N², so six-figure counts run at full rate.
    count: 131072,
    steps: 1,
    paused: false,
    totalMass: 40960,
    // Gravity sized so the *slow* meditative speeds are roughly circular:
    // clusters form and shear apart instead of everything raining into one
    // clump at the pole.
    gravity: 0.0000001,
    // Small timestep + single-ish steps keep world-space motion slow and
    // meditative now that sim units span an 84 m dome.
    dt: 0.008,
    softening: 0.08,
    // Barnes-Hut acceptance: cellWidth² < θ²·distance². Lower = more exact.
    theta: 0.8,
    // The post's pairing: dome spring on, gentle damping so the shell
    // settles instead of ringing forever.
    damping: 0.999,
    maxSpeed: 0.3,
    shellK: 6,
    shellR: SIM_R,
    floorK: 4,
    spin: 1.0,
    dispersion: 0.04,
    // Perpetual gentle motion: curl-ish noise stirring + azimuthal breeze
    // that offsets damping so the dome never freezes or fully collapses.
    flow: 0.04,
    // Keep the breeze tiny: its equilibrium speed against damping goes
    // centrifugal and drains particles to the rim if pushed much higher.
    swirl: 0.004,
    massMin: 1,
    massMax: 3,
    pointerMode: 'attract',
    strength: 0.35,
    pointerSoftening: 0.08,
    pointerMassScale: 1.0,
    sizeScale: 0.004,
    minSize: 0.002,
    // Speeds are normalized by maxSpeed before coloring; 2.5 tops the ramp
    // out at ~40% of the cap. Combined with the squared response curve,
    // ambient drift (~15-20% of cap) stays blue while attracted particles
    // shoot through the pink midrange into orange.
    colorScale: 2.5,
    colorLow: '#1b3cff',
    colorHigh: '#ff4d2e',
    brightness: 1.5,
  };

  // Seat the dome's equator just below the lowest terrain around its rim so
  // the floor spring keeps every star above ground.
  const { x: cx, z: cz, r: R } = DOME;
  let baseY = Infinity;
  for (let i = 0; i < 48; i++) {
    const a = (i / 48) * Math.PI * 2;
    baseY = Math.min(baseY, terrainHeight(cx + Math.cos(a) * R * 0.95, cz + Math.sin(a) * R * 0.95));
  }
  baseY -= 0.25;
  const center = new THREE.Vector3(cx, baseY, cz);

  const uMaxSpeed = uniform(params.maxSpeed);
  const uSizeScale = uniform(params.sizeScale);
  const uMinSize = uniform(params.minSize);
  const uColorScale = uniform(params.colorScale);
  const uColLow = uniform(new THREE.Color(params.colorLow));
  const uColHigh = uniform(new THREE.Color(params.colorHigh));
  const uBrightness = uniform(params.brightness);
  const uCenter = uniform(center.clone());

  // Archimedes seeding: uniform height + random angle lands uniformly on the
  // hemisphere. Swirl about the vertical axis, faster near the rim.
  function generateInitData(n: number) {
    const pos = new Float32Array(n * 4);
    const vel = new Float32Array(n * 4);
    let total = 0;
    for (let i = 0; i < n; i++) {
      const m = params.massMin + Math.random() * Math.max(0, params.massMax - params.massMin);
      pos[i * 4 + 3] = m;
      total += m;
    }
    const norm = params.totalMass / total;
    for (let i = 0; i < n; i++) pos[i * 4 + 3] *= norm;

    const Rs = params.shellR;
    for (let i = 0; i < n; i++) {
      const y = Math.random();
      const rho = Math.sqrt(Math.max(0, 1 - y * y));
      const a = Math.random() * Math.PI * 2;
      pos[i * 4 + 0] = Math.cos(a) * rho * Rs;
      pos[i * 4 + 1] = y * Rs;
      pos[i * 4 + 2] = Math.sin(a) * rho * Rs;
      const ringR = rho * Rs;
      const vCirc = Math.sqrt((params.gravity * params.totalMass * 0.5) / Math.sqrt(ringR * ringR + 0.05 * 0.05));
      const v = params.spin * vCirc * rho;
      vel[i * 4 + 0] = Math.sin(a) * v + randn() * params.dispersion * vCirc;
      vel[i * 4 + 1] = randn() * 0.04 * vCirc * (1 + params.dispersion);
      vel[i * 4 + 2] = -Math.cos(a) * v + randn() * params.dispersion * vCirc;
    }
    return { pos, vel };
  }

  let n = params.count;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let posArr: any, velArr: any;
  let posBuf: GPUBuffer, velBuf: GPUBuffer;
  let solver: OctreeDomeSolver;
  let sprite: THREE.Sprite | null = null;
  let simTime = 0;

  // Gravity is the blog post's Barnes-Hut octree pyramid, running as raw
  // WebGPU compute on the renderer's own device. Three owns the pos/vel
  // buffers (the sprite material reads them as instanced vertex data); the
  // solver binds the same GPUBuffers into its pipelines, so physics and
  // pixels share one memory and nothing is ever copied.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const backend = (renderer as any).backend;
  if (!backend?.device) {
    throw new Error('NBodyDome needs the WebGPU backend (renderer fell back to WebGL)');
  }
  const device: GPUDevice = backend.device;

  function build() {
    n = params.count;
    const init = generateInitData(n);
    posArr = instancedArray(init.pos, 'vec4');
    velArr = instancedArray(init.vel, 'vec4');

    // No-op compute over both arrays so the backend allocates their
    // GPUBuffers with STORAGE usage; the solver then binds those directly.
    const touch = Fn(() => {
      posArr.element(instanceIndex).assign(posArr.element(instanceIndex));
      velArr.element(instanceIndex).assign(velArr.element(instanceIndex));
    })().compute(n);
    renderer.compute(touch);
    posBuf = backend.get(posArr.value).buffer;
    velBuf = backend.get(velArr.value).buffer;
    solver = new OctreeDomeSolver(device, n, posBuf, velBuf);

    const material = new THREE.SpriteNodeMaterial();
    const p = posArr.toAttribute();
    const v = velArr.toAttribute();
    material.positionNode = uCenter.add(p.xyz.mul(SCALE));
    const rad = uMinSize.max(uSizeScale.mul(p.w.pow(1 / 3)));
    material.scaleNode = rad.mul(2 * SCALE);
    // Color by speed as a fraction of the speed cap so the low->high ramp
    // always spans the actual speed range, regardless of how chill the
    // physics defaults get. The squared response keeps ambient drift pinned
    // near the slow color so pointer-stirred particles pop visibly orange.
    const speed = v.xyz.length();
    const t = speed.div(uMaxSpeed.max(1e-6)).mul(uColorScale).saturate().pow(2);
    const col = mix(uColLow, uColHigh, t);
    const d = uv().sub(0.5).length().mul(2);
    const a = smoothstep(1, 0, d);
    material.colorNode = vec4(col.mul(a).mul(uBrightness), a);
    material.transparent = true;
    material.blending = THREE.AdditiveBlending;
    material.depthWrite = false;
    material.depthTest = true;
    material.fog = false;

    sprite = new THREE.Sprite(material);
    sprite.count = n;
    sprite.frustumCulled = false;
    scene.add(sprite);
  }

  function teardown() {
    if (!sprite) return;
    scene.remove(sprite);
    sprite.material.dispose();
    sprite = null;
    solver.dispose();
    posBuf.destroy();
    velBuf.destroy();
  }

  function reseed() {
    const init = generateInitData(n);
    device.queue.writeBuffer(posBuf, 0, init.pos);
    device.queue.writeBuffer(velBuf, 0, init.vel);
  }

  build();

  // ---------------- forcers ----------------

  // Per-slot forcer state, uploaded into the solver's uniforms each frame:
  // x, y, z = sim-space position, w = gain.
  const forcers: [number, number, number, number][] =
    Array.from({ length: FORCER_SLOTS }, () => [0, 0, 0, 0]);

  function clearForcers() {
    for (const f of forcers) f[3] = 0;
  }

  function setForcer(slot: number, simPoint: THREE.Vector3, sign: number) {
    const f = forcers[slot];
    f[0] = simPoint.x;
    f[1] = simPoint.y;
    f[2] = simPoint.z;
    f[3] = sign * params.strength * params.gravity * params.totalMass * params.pointerMassScale;
  }

  const oc = new THREE.Vector3();
  function intersectRay(origin: THREE.Vector3, dir: THREE.Vector3): THREE.Vector3 | null {
    const Rw = params.shellR * SCALE;
    oc.copy(origin).sub(center);
    const b = oc.dot(dir);
    const c = oc.lengthSq() - Rw * Rw;
    const disc = b * b - c;
    if (disc <= 0) return null;
    const sq = Math.sqrt(disc);
    // Nearest hit on the upper hemisphere; fall through to the far side so
    // pointing across the dome from outside still works.
    for (const tHit of [-b - sq, -b + sq]) {
      if (tHit <= 0) continue;
      const hit = new THREE.Vector3().copy(dir).multiplyScalar(tHit).add(origin).sub(center).divideScalar(SCALE);
      if (hit.y > -0.05 * params.shellR) return hit;
    }
    return null;
  }

  function simToWorld(simPoint: THREE.Vector3, out = new THREE.Vector3()): THREE.Vector3 {
    return out.copy(simPoint).multiplyScalar(SCALE).add(center);
  }

  // ---------------- tweakpane ----------------

  // tweakpane v4 typings omit folder helpers used at runtime
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const pane = new Pane({ title: 'N-Body Dome' }) as any;
  const paneContainer = pane.element.parentElement!;
  paneContainer.style.zIndex = '20';
  paneContainer.style.maxHeight = 'calc(100vh - 16px)';
  paneContainer.style.overflowY = 'auto';
  paneContainer.style.overscrollBehavior = 'contain';

  const fSim = pane.addFolder({ title: 'Simulation' });
  fSim.addBinding(params, 'count', {
    options: {
      '4k': 4096,
      '8k': 8192,
      '16k': 16384,
      '32k': 32768,
      '64k': 65536,
      '128k': 131072,
      '256k': 262144,
      '512k': 524288,
      '1M': 1048576,
    },
  }).on('change', () => { teardown(); build(); });
  fSim.addBinding(params, 'steps', { min: 1, max: 8, step: 1, label: 'steps / frame' });
  fSim.addBinding(params, 'paused');
  fSim.addButton({ title: 'reseed (r)' }).on('click', reseed);

  const fPhys = pane.addFolder({ title: 'Gravity & Integration' });
  fPhys.addBinding(params, 'totalMass', { min: 1000, max: 100000, step: 500, label: 'total mass' });
  fPhys.addBinding(params, 'gravity', { min: 0, max: 0.000002, step: 0.00000002 });
  fPhys.addBinding(params, 'dt', { min: 0.001, max: 0.05, step: 0.0005, label: 'timestep' });
  fPhys.addBinding(params, 'softening', { min: 0.001, max: 0.5, step: 0.001, label: 'softening ε' });
  fPhys.addBinding(params, 'theta', { min: 0.4, max: 1.4, step: 0.05, label: 'tree θ' });
  fPhys.addBinding(params, 'damping', { min: 0.9, max: 1.0, step: 0.0005 });
  fPhys.addBinding(params, 'maxSpeed', { min: 0.05, max: 5.0, step: 0.05 });

  const fDome = pane.addFolder({ title: 'Dome Constraint' });
  fDome.addBinding(params, 'shellK', { min: 0, max: 20, step: 0.5, label: 'dome strength' });
  fDome.addBinding(params, 'shellR', { min: 0.4, max: 1.1, step: 0.01, label: 'shell radius' });
  fDome.addBinding(params, 'floorK', { min: 0, max: 8, step: 0.5, label: 'floor stiffness ×' });

  const fFlow = pane.addFolder({ title: 'Flow & Wind' });
  fFlow.addBinding(params, 'flow', { min: 0, max: 0.15, step: 0.001, label: 'noise stir' });
  fFlow.addBinding(params, 'swirl', { min: 0, max: 0.08, step: 0.001, label: 'breeze' });

  const fSeed = pane.addFolder({ title: 'Initial Conditions' });
  fSeed.addBinding(params, 'spin', { min: 0, max: 2.0, step: 0.01, label: 'orbital spin' });
  fSeed.addBinding(params, 'dispersion', { min: 0, max: 1.0, step: 0.01, label: 'velocity noise' });
  fSeed.addBinding(params, 'massMin', { min: 0.1, max: 10, step: 0.1, label: 'mass min' });
  fSeed.addBinding(params, 'massMax', { min: 0.1, max: 20, step: 0.1, label: 'mass max' });

  const fPtr = pane.addFolder({ title: 'Pointer & Controllers' });
  fPtr.addBinding(params, 'pointerMode', {
    label: 'mouse mode (t)',
    options: { attract: 'attract', repulse: 'repulse' },
  });
  fPtr.addBinding(params, 'strength', { min: 0, max: 16, step: 0.1, label: 'force strength' });
  fPtr.addBinding(params, 'pointerSoftening', { min: 0.001, max: 0.5, step: 0.001, label: 'pointer ε' });
  fPtr.addBinding(params, 'pointerMassScale', { min: 0, max: 4, step: 0.05, label: 'mass coupling' });

  const fLook = pane.addFolder({ title: 'Particle Appearance' });
  fLook.addBinding(params, 'sizeScale', { min: 0, max: 0.02, step: 0.0001, label: 'size ∝ mass^⅓' });
  fLook.addBinding(params, 'minSize', { min: 0, max: 0.01, step: 0.0001 });
  fLook.addBinding(params, 'colorScale', { min: 0, max: 6, step: 0.01, label: 'speed → color (×max)' });
  fLook.addBinding(params, 'colorLow', { label: 'slow color' });
  fLook.addBinding(params, 'colorHigh', { label: 'fast color' });
  fLook.addBinding(params, 'brightness', { min: 0, max: 6, step: 0.05 });

  window.addEventListener('keydown', (e) => {
    if (e.code === 'KeyT') {
      params.pointerMode = params.pointerMode === 'attract' ? 'repulse' : 'attract';
      pane.refresh();
    } else if (e.code === 'KeyR') {
      reseed();
    } else if (e.code === 'KeyP') {
      params.paused = !params.paused;
      pane.refresh();
    }
  });

  function update() {
    if (params.paused) return;
    uMaxSpeed.value = params.maxSpeed;
    uSizeScale.value = params.sizeScale;
    uMinSize.value = params.minSize;
    uColorScale.value = params.colorScale;
    uColLow.value.set(params.colorLow);
    uColHigh.value.set(params.colorHigh);
    uBrightness.value = params.brightness;

    simTime += params.dt * params.steps;
    solver.writeParams({
      count: n,
      dt: params.dt,
      gravity: params.gravity,
      softening: params.softening,
      theta: params.theta,
      damping: params.damping,
      shellR: params.shellR,
      shellK: params.shellK,
      floorK: params.floorK,
      maxSpeed: params.maxSpeed,
      flow: params.flow,
      swirl: params.swirl,
      time: simTime,
      pointerSoftening: params.pointerSoftening,
      forcers,
    });

    // Whole sim — tree rebuild + force/integrate — in one compute pass per
    // substep, on the same queue Three renders from, so ordering is free.
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    for (let k = 0; k < params.steps; k++) solver.encode(pass);
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  return { update, reseed, clearForcers, setForcer, intersectRay, simToWorld, center, params, pane };
}
