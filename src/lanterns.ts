import * as THREE from 'three/webgpu';
import {
  Fn, If, Loop, Break, uniform, uniformArray, wgslFn,
  vec2, vec3, vec4, float, mix, smoothstep,
  cameraPosition, positionWorld, normalView,
} from 'three/tsl';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { terrainHeight } from './terrain';

// Two lava-lamp lanterns flanking the spawn meadow, each a vintage lantern
// body (GLB) hanging from a shepherd's-hook post. Built from the "Light in a
// Browser Tab" blog series and its 3D sibling, the standalone LavaLamp repo:
//
//  - The wax is LavaLamp's blob sim (src/lib/blobSim.ts), ported intact:
//    a molten pool plus a dozen free blobs with temperature. Heat soaks in
//    above the coil, buoyancy lifts what warmed past neutral, the throat
//    chills it, and it sinks home to remelt. Nobody tells it to circulate.
//  - The look is LavaLamp's raymarcher (src/shaders/lamp3d.wgsl), rebuilt
//    in TSL on the vessel's own surface mesh: exponential smooth-min welds
//    the blobs into one gooey field, the surface emits by temperature, a
//    glowing coil torus flickers at the base, an analytic coil light models
//    the underglow, and volumetric glow accumulates along the interior ray
//    so the liquid itself shines.
//  - The bonfire post's thesis — "everything is a light" — is honoured the
//    only way a rasterizer can afford: each lantern carries a real
//    PointLight, so the wax genuinely lights the grass and its own cage.
//
// Helper WGSL (vessel profile, palette) is shared with the TSL graph via
// wgslFn, so the lanterns speak both languages at once.

// ---- vessel + sim geometry (lamp units, mirrored from LavaLamp) ---------------
const GLASS_BOT = -0.34;
const GLASS_TOP = 0.6;
const GLASS_THICK = 0.012;
const LAVA_BOT = -0.3;
const LAVA_TOP = 0.5;
const HEATER_Y = -0.12;
const COIL_Y = -0.27;
const COIL_R = 0.115;
const SCALE = 0.675; // lamp units → metres

// Glass interior radius at height y: a teardrop — soft bulge low, slim throat.
function vesselR(y: number): number {
  const t = Math.min(Math.max((y - GLASS_BOT) / (GLASS_TOP - GLASS_BOT), 0), 1);
  const s = Math.min(Math.max(t * 0.92 + 0.04, 0), 1);
  const ss = s * s * (3 - 2 * s);
  const taper = 0.205 + (0.087 - 0.205) * ss;
  const bulge = 0.045 * Math.sin(Math.min(t * 2.2, 1) * Math.PI * 0.5) * (1 - t * t);
  return taper + bulge;
}

// ---- blob sim (LavaLamp's BlobSim, verbatim dynamics) ----------------------------

const clamp01 = (v: number, lo: number, hi: number): number => Math.min(Math.max(v, lo), hi);
const smoothJs = (a: number, b: number, x: number): number => {
  const t = clamp01((x - a) / (b - a), 0, 1);
  return t * t * (3 - 2 * t);
};

interface Blob {
  x: number; y: number; z: number;
  vx: number; vy: number; vz: number;
  r: number;
  temp: number;
  seed: number;
}

const FREE = 12;
// The molten pool: three squashed spheres parked on the floor of the glass.
const POOL: Array<{ x: number; z: number; r: number }> = [
  { x: -0.08, z: 0.02, r: 0.105 },
  { x: 0.0, z: -0.01, r: 0.135 },
  { x: 0.08, z: 0.0, r: 0.105 },
];
const POOL_Y = -0.315;
const POOL_SQUASH = 0.55;
const BLOB_COUNT = POOL.length + FREE;

class BlobSim {
  poolTemp = 0.3;
  private blobs: Blob[] = [];
  private t = 0;

  constructor() { this.reset(); }

  reset(): void {
    this.poolTemp = 0.3;
    this.t = 0;
    this.blobs = [];
    for (let i = 0; i < FREE; i++) {
      const r = 0.048 + Math.random() * 0.038;
      const y = LAVA_BOT + 0.08 + Math.random() * 0.55;
      const maxR = Math.max(vesselR(y) - r - 0.03, 0.01);
      const a = Math.random() * Math.PI * 2;
      const rad = Math.sqrt(Math.random()) * maxR;
      this.blobs.push({
        x: Math.cos(a) * rad, y, z: Math.sin(a) * rad,
        vx: 0, vy: 0, vz: 0,
        r,
        temp: 0.2 + Math.random() * 0.5,
        seed: Math.random() * 100,
      });
    }
  }

  step(dt: number, heat: number, buoy: number): void {
    this.t += dt;
    this.poolTemp += (clamp01(heat * 0.55, 0, 1.05) - this.poolTemp) * dt * 0.5;

    const bs = this.blobs;
    // gentle pair repulsion so the lava stays plural instead of one megablob
    for (let i = 0; i < bs.length; i++) {
      for (let j = i + 1; j < bs.length; j++) {
        const a = bs[i], b = bs[j];
        const dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
        const d = Math.hypot(dx, dy, dz);
        const minD = (a.r + b.r) * 0.75;
        if (d < minD && d > 1e-5) {
          const push = ((minD - d) / minD) * 0.55 * dt;
          const ux = dx / d, uy = dy / d, uz = dz / d;
          a.vx -= ux * push; a.vy -= uy * push; a.vz -= uz * push;
          b.vx += ux * push; b.vy += uy * push; b.vz += uz * push;
        }
      }
    }

    for (const b of bs) {
      // --- temperature -------------------------------------------------
      if (b.y < HEATER_Y) {
        const f = clamp01((HEATER_Y - b.y) / (HEATER_Y - LAVA_BOT), 0, 1);
        b.temp += heat * 0.5 * dt * f;
        b.temp += (this.poolTemp - b.temp) * dt * 0.8 * f; // contact with the pool
      }
      const topF = smoothJs(LAVA_TOP - 0.25, LAVA_TOP, b.y);
      b.temp -= 0.8 * b.temp * (0.05 + 0.3 * topF) * dt;
      b.temp = clamp01(b.temp, 0.02, 1.15);

      // --- forces -------------------------------------------------------
      const wander = 0.05;
      b.vx += wander * Math.sin(this.t * 0.5 + b.seed * 7.3) * dt;
      b.vz += wander * Math.cos(this.t * 0.43 + b.seed * 9.1) * dt;
      b.vy += buoy * 0.16 * (b.temp - 0.46) * dt;
      // cold wax near the floor drifts toward the axis, back over the coil
      if (b.y < HEATER_Y + 0.1) {
        b.vx -= b.x * 0.25 * dt;
        b.vz -= b.z * 0.25 * dt;
      }

      const drag = Math.exp(-1.5 * dt);
      b.vx *= drag; b.vy *= drag; b.vz *= drag;
      const sp = Math.hypot(b.vx, b.vy, b.vz);
      if (sp > 0.45) {
        const s = 0.45 / sp;
        b.vx *= s; b.vy *= s; b.vz *= s;
      }
      b.x += b.vx * dt; b.y += b.vy * dt; b.z += b.vz * dt;

      // --- containment ----------------------------------------------------
      const maxRad = Math.max(vesselR(b.y) - b.r * 0.8 - 0.02, 0.012);
      const rad = Math.hypot(b.x, b.z);
      if (rad > maxRad) {
        const s = maxRad / rad;
        b.x *= s; b.z *= s;
        const vr = (b.vx * b.x + b.vz * b.z) / Math.max(rad * s, 1e-5);
        if (vr > 0) {
          b.vx -= (b.x / (rad * s)) * vr;
          b.vz -= (b.z / (rad * s)) * vr;
        }
      }
      const minY = LAVA_BOT + b.r * 0.4;
      const maxY = LAVA_TOP - b.r;
      if (b.y < minY) { b.y += (minY - b.y) * clamp01(6 * dt, 0, 1); b.vy *= 0.9; }
      if (b.y > maxY) { b.y += (maxY - b.y) * clamp01(6 * dt, 0, 1); b.vy *= 0.9; }
    }
  }

  // Pack into the uniform array: [0..N) = (pos, radius), [N..2N) = (temp, squash).
  fill(out: THREE.Vector4[]): void {
    let i = 0;
    for (const p of POOL) {
      out[i].set(p.x, POOL_Y, p.z, p.r);
      out[i + BLOB_COUNT].set(this.poolTemp, POOL_SQUASH, 0, 0);
      i++;
    }
    for (const b of this.blobs) {
      // hot wax swells a touch — reads as thermal expansion
      out[i].set(b.x, b.y, b.z, b.r * (1 + 0.15 * clamp01(b.temp, 0, 1)));
      out[i + BLOB_COUNT].set(b.temp, 1, 0, 0);
      i++;
    }
  }
}

// ---- WGSL helpers shared with the TSL graph --------------------------------------

// wgslFn's typings return a bare Node with no operator methods; calls are fed
// back through float()/vec3() at the use site, so loosen the callable type.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type WgslNodeFn = (...args: unknown[]) => any;

// The vessel profile, mirrored from vesselR() above (single source of truth
// for the CPU sim, the lathe geometry, and the raymarch wall clamp).
const vesselRW = wgslFn(`
  fn vesselRW(y: f32) -> f32 {
    let t = clamp((y + 0.34) / 0.94, 0.0, 1.0);
    let s = clamp(t * 0.92 + 0.04, 0.0, 1.0);
    let ss = s * s * (3.0 - 2.0 * s);
    let taper = mix(0.205, 0.087, ss);
    let bulge = 0.045 * sin(min(t * 2.2, 1.0) * 1.5707963) * (1.0 - t * t);
    return taper + bulge;
  }
`) as unknown as WgslNodeFn;

// lamp3d.wgsl's waxColor ramp, parameterised by tint so the two lanterns can
// disagree about what "hot" means.
const waxRamp = wgslFn(`
  fn waxRamp(t: f32, deep: vec3<f32>, hot: vec3<f32>) -> vec3<f32> {
    let c0 = deep * 0.06; // cold: a near-black shade of the tint
    let c3 = mix(hot, vec3<f32>(1.0), 0.65);
    let a = smoothstep(0.05, 0.45, t);
    let b = smoothstep(0.40, 0.75, t);
    let c = smoothstep(0.70, 1.05, t);
    return mix(mix(mix(c0, deep, a), hot, b), c3, c);
  }
`) as unknown as WgslNodeFn;

// ---- public interface ---------------------------------------------------------------

interface LanternSpec {
  position: THREE.Vector3; // ground anchor, world (the lamp hangs above it)
  deep: THREE.Color;       // cold/deep tint of the wax ramp
  hot: THREE.Color;        // hot tint of the wax ramp
  light: THREE.Color;
  phase: number;           // desynchronises the light flicker
}

export interface Lanterns {
  group: THREE.Group;
  params: {
    scale: number;
    heat: number; buoyancy: number; goo: number;
    glow: number; lightIntensity: number;
  };
  attachPane: (pane: { addFolder(o: { title: string }): { addBinding(o: object, k: string, opt?: object): unknown } }) => void;
  update: (dt: number) => void;
}

// ---- the vintage lantern body (GLB) -------------------------------------------------------

// Brass, still used for the post hardware.
const brass = new THREE.MeshStandardNodeMaterial({
  color: 0x9c7a3c, metalness: 0.9, roughness: 0.32,
});

// The Tripo-generated model is ~0.98 units tall with its glass globe spanning
// y ∈ [-0.22, 0.26], radius 0.145 ("tripo_part_0"). At this scale the globe
// cavity matches the raymarched vessel (y ∈ [-0.153, 0.27], r ≤ 0.11 m)
// almost exactly, so the lava vessel simply replaces the baked-in glass.
const MODEL_URL = '/lantern.glb';
const MODEL_SCALE = 1.32;
const MODEL_LIFT = 0.062;     // aligns the globe cavity with the wax vessel
const MODEL_GLASS = 'tripo_part_0';
const MODEL_HANG = 0.705;     // hook height above the vessel centre

// ---- shepherd's-hook post -----------------------------------------------------------------

// Builds a wooden post with a brass hook arm, in lantern-local space: the
// ground is y = 0 and the hook tip (where the hanging ring sits) is `hookY`
// straight above the origin. The post stands `reach` metres away along
// `outward`. The whole thing lives under the lantern's root group, so the
// scale slider grows it with the lamp.
function buildPost(hookY: number, outward: THREE.Vector3): THREE.Group {
  const post = new THREE.Group();
  const wood = new THREE.MeshStandardNodeMaterial({
    color: 0x4a3726, metalness: 0, roughness: 0.95,
  });

  const reach = 0.58;
  const baseX = outward.x * reach;
  const baseZ = outward.z * reach;
  const postTop = hookY + 0.4;

  // the post itself, sunk into the ground
  const height = postTop + 0.4;
  const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.062, 0.075, height, 10), wood);
  trunk.position.set(baseX, height / 2 - 0.4, baseZ);
  trunk.castShadow = true;
  post.add(trunk);

  // a small collar where the arm meets the post
  const collar = new THREE.Mesh(new THREE.TorusGeometry(0.075, 0.016, 8, 16), brass);
  collar.position.set(baseX, postTop - 0.28, baseZ);
  post.add(collar);

  // brass hook arm: out from the post top, arcing over to dip through the ring
  const inX = -outward.x, inZ = -outward.z;
  const arm = new THREE.CatmullRomCurve3([
    new THREE.Vector3(baseX, postTop - 0.32, baseZ),
    new THREE.Vector3(baseX, postTop - 0.06, baseZ),
    new THREE.Vector3(baseX + inX * reach * 0.45, postTop + 0.05, baseZ + inZ * reach * 0.45),
    new THREE.Vector3(baseX + inX * reach * 0.9, postTop - 0.025, baseZ + inZ * reach * 0.9),
    new THREE.Vector3(0, hookY + 0.12, 0),
    new THREE.Vector3(0, hookY - 0.02, 0),
  ]);
  const armMesh = new THREE.Mesh(new THREE.TubeGeometry(arm, 32, 0.019, 8), brass);
  armMesh.castShadow = true;
  post.add(armMesh);

  return post;
}

// ---- the lanterns --------------------------------------------------------------------------

export function createLanterns(
  scene: THREE.Scene,
  positions: [THREE.Vector3, THREE.Vector3],
): Lanterns {
  const HANG = 1.7; // vessel centre above the terrain, before the scale slider

  // spec positions anchor each lantern's root group to the ground; the
  // vessel hangs HANG·scale above it
  const specs: LanternSpec[] = [
    {
      // ember lantern: the lava post's palette (and the repulse hand)
      position: positions[0].clone().setY(terrainHeight(positions[0].x, positions[0].z)),
      deep: new THREE.Color(0x8c0f08),
      hot: new THREE.Color(0xffa14f),
      light: new THREE.Color(0xff9347),
      phase: 0.0,
    },
    {
      // spirit lantern: the attract hand's cyan, wax from a colder star
      position: positions[1].clone().setY(terrainHeight(positions[1].x, positions[1].z)),
      deep: new THREE.Color(0x0a4a66),
      hot: new THREE.Color(0x4fd8ff),
      light: new THREE.Color(0x4fd8ff),
      phase: 2.4,
    },
  ];

  const params = {
    scale: 1.6,
    heat: 0.95, buoyancy: 5.2, goo: 0.015,
    glow: 0.9, lightIntensity: 5.0,
  };

  const group = new THREE.Group();
  scene.add(group);

  // ---- shared uniforms ----
  // vessel centres in world space, refreshed whenever the scale changes
  const uPosA = uniform(specs[0].position.clone().add(new THREE.Vector3(0, HANG * params.scale, 0)));
  const uPosB = uniform(specs[1].position.clone().add(new THREE.Vector3(0, HANG * params.scale, 0)));
  const uTime = uniform(0);
  const uGlow = uniform(params.glow);
  const uHeat = uniform(params.heat);
  const uGoo = uniform(params.goo);
  const uScale = uniform(params.scale);

  const tintDeepA = vec3(specs[0].deep.r, specs[0].deep.g, specs[0].deep.b);
  const tintDeepB = vec3(specs[1].deep.r, specs[1].deep.g, specs[1].deep.b);
  const tintHotA = vec3(specs[0].hot.r, specs[0].hot.g, specs[0].hot.b);
  const tintHotB = vec3(specs[1].hot.r, specs[1].hot.g, specs[1].hot.b);

  // exponential smooth-max (mirror of lamp3d.wgsl's smax), as a node helper
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const smax = (a: any, b: any, k: number) => {
    const h = float(0.5).sub(a.sub(b).div(2 * k)).clamp(0, 1);
    return mix(a, b, h).add(h.mul(float(1).sub(h)).mul(k));
  };

  // ---- raymarched wax: one vessel mesh per lantern -------------------------------------
  const vesselProfile: THREE.Vector2[] = [];
  for (let i = 0; i <= 28; i++) {
    const lampY = GLASS_BOT + (GLASS_TOP - GLASS_BOT) * (i / 28);
    vesselProfile.push(new THREE.Vector2((vesselR(lampY) + GLASS_THICK) * SCALE, lampY * SCALE));
  }
  const vesselGeo = new THREE.LatheGeometry(vesselProfile, 48);

  interface WaxRig { sim: BlobSim; blobVecs: THREE.Vector4[]; mesh: THREE.Mesh }
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const makeWax = (spec: LanternSpec, uPos: any, deep: any, hot: any): WaxRig => {
    const sim = new BlobSim();
    const blobVecs = Array.from({ length: BLOB_COUNT * 2 }, () => new THREE.Vector4());
    sim.fill(blobVecs);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const uBlobs = uniformArray(blobVecs) as any;

    // lamp3d.wgsl's lavaField: exponential smooth-min over every blob with a
    // weight-blended temperature, wobble, and the glass wall clamp.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const lavaField = Fn(([p]: any[]) => {
      const sum = float(0).toVar();
      const tsum = float(0).toVar();
      Loop(BLOB_COUNT, ({ i }) => {
        const posr = uBlobs.element(i);
        const info = uBlobs.element(i.add(BLOB_COUNT));
        const q = p.sub(posr.xyz).toVar();
        q.y.assign(q.y.div(info.y));
        const di = q.length().sub(posr.w).mul(info.y.min(1));
        const w = di.div(uGoo.max(0.02)).negate().exp();
        sum.addAssign(w);
        tsum.addAssign(w.mul(info.x));
      });
      const d = uGoo.max(0.02).mul(sum.max(1e-7).log()).negate().toVar();
      const temp = tsum.div(sum.max(1e-7));
      // organic wobble, stronger on hot wax
      d.addAssign(
        p.y.mul(14).add(uTime.mul(1.1)).sin()
          .mul(p.x.mul(12).sub(uTime.mul(0.7)).sin())
          .mul(p.z.mul(13).add(uTime.mul(0.9)).sin())
          .mul(0.01).mul(temp.clamp(0, 1).mul(0.5).add(0.5)),
      );
      // squash against the glass interior and the vessel floor — held a little
      // off the glass so a rim of dark liquid keeps the blobs reading as
      // shapes instead of a wash
      const wall = p.xz.length().sub(float(vesselRW(p.y)).sub(0.022));
      const d2 = smax(smax(d, wall, 0.022), float(GLASS_BOT + 0.008).sub(p.y), 0.02);
      return vec2(d2, temp);
    });

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const lum = (t: any) => {
      const g = smoothstep(0.14, 0.95, t);
      return g.mul(g).mul(2.6).add(0.004);
    };

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const shadeWax = Fn(([p, rd, temp]: any[]) => {
      const e = 0.0035;
      const n = vec3(
        lavaField(p.add(vec3(e, 0, 0))).x.sub(lavaField(p.sub(vec3(e, 0, 0))).x),
        lavaField(p.add(vec3(0, e, 0))).x.sub(lavaField(p.sub(vec3(0, e, 0))).x),
        lavaField(p.add(vec3(0, 0, e))).x.sub(lavaField(p.sub(vec3(0, 0, e))).x),
      ).normalize().toVar();
      const col = vec3(waxRamp(temp, deep, hot)).toVar();
      const emis = col.mul(lum(temp)).mul(uGlow);
      // the coil lights cold wax from below
      const lv = vec3(0, COIL_Y, 0).sub(p).toVar();
      const ld = lv.length();
      const L = lv.div(ld.max(1e-3));
      const coilLight = hot.mul(n.dot(L).max(0)).mul(uHeat.mul(0.6)).div(ld.mul(ld).mul(18).add(1));
      const albedo = col.mul(0.5).add(deep.mul(0.05));
      const fres = float(1).sub(n.dot(rd.negate()).max(0)).pow(3);
      const rim = mix(deep, hot, 0.4).mul(temp.add(0.25)).mul(fres).mul(0.8).mul(uGlow);
      // a cool key-light specular so blob tops read glossy, like real wax
      const KL = vec3(0.55, 0.65, 0.35).normalize();
      const spec = rd.reflect(n).dot(KL).max(0).pow(30).mul(0.35);
      return emis.add(albedo.mul(coilLight)).add(rim).add(vec3(spec));
    });

    const mat = new THREE.MeshBasicNodeMaterial();
    mat.colorNode = Fn(() => {
      // march in lamp units from the fragment on the vessel surface
      const rd = positionWorld.sub(cameraPosition).normalize().toVar();
      const ro = positionWorld.sub(uPos).div(uScale.mul(SCALE)).toVar();
      const liquid = deep.mul(0.025).add(vec3(0.002, 0.002, 0.004));
      const out = liquid.toVar();
      const glowAcc = vec3(0).toVar();
      const t = float(0.02).toVar();
      Loop(64, () => {
        const p = ro.add(rd.mul(t)).toVar();
        const lf = lavaField(p).toVar();
        const cq = vec2(p.xz.length().sub(COIL_R), p.y.sub(COIL_Y));
        const dCoil = cq.length().sub(0.02).toVar();
        const d = lf.x.min(dCoil);
        const stepLen = d.mul(0.85).clamp(0.005, 0.07).toVar();
        // volumetric glow: hot wax and the coil bleed light into the liquid
        glowAcc.addAssign(
          vec3(waxRamp(lf.y, deep, hot)).mul(lum(lf.y))
            .mul(lf.x.max(0).mul(-24).exp()).mul(stepLen).mul(1.5),
        );
        glowAcc.addAssign(hot.mul(dCoil.max(0).mul(-26).exp()).mul(stepLen).mul(5).mul(uHeat));
        If(d.lessThan(0.0018), () => {
          If(dCoil.lessThan(lf.x), () => {
            const flicker = uTime.mul(6).add(p.x.mul(30)).sin().mul(0.1).add(0.9);
            out.assign(hot.mul(5).mul(uHeat).mul(flicker));
          }).Else(() => {
            out.assign(vec3(shadeWax(p, rd, lf.y)));
          });
          Break();
        });
        t.addAssign(stepLen);
        If(t.greaterThan(1.9), () => { Break(); });
      });
      // faint cool sheen on the glass itself
      const sheen = float(1).sub(normalView.z.abs()).pow(3).mul(0.10);
      return vec4(out.add(glowAcc.mul(uGlow)).add(vec3(0.5, 0.7, 0.9).mul(sheen)), 1);
    })();
    mat.fog = true;

    const mesh = new THREE.Mesh(vesselGeo, mat);
    return { sim, blobVecs, mesh };
  };

  const waxA = makeWax(specs[0], uPosA, tintDeepA, tintHotA);
  const waxB = makeWax(specs[1], uPosB, tintDeepB, tintHotB);
  // arrive mid-churn instead of cold
  for (let i = 0; i < 700; i++) {
    waxA.sim.step(1 / 30, params.heat, params.buoyancy);
    waxB.sim.step(1 / 30, params.heat, params.buoyancy);
  }

  // ---- lantern roots: body, post, light, wax all under one scalable group ----------------------
  // Each post stands on the far side of its lantern, so the pair frames the
  // player's view instead of crowding it.
  const apart = specs[1].position.clone().sub(specs[0].position).setY(0).normalize();

  interface LanternRig {
    root: THREE.Group;
    light: THREE.PointLight;
    wax: WaxRig;
    posUniform: { value: THREE.Vector3 };
    phase: number;
  }
  const rigs: LanternRig[] = specs.map((spec, idx) => {
    const root = new THREE.Group();
    root.position.copy(spec.position); // ground anchor
    root.scale.setScalar(params.scale);
    group.add(root);

    const wax = idx === 0 ? waxA : waxB;
    wax.mesh.position.y = HANG;
    root.add(wax.mesh);

    const outward = idx === 0 ? apart.clone().negate() : apart.clone();
    root.add(buildPost(HANG + MODEL_HANG, outward));

    const light = new THREE.PointLight(spec.light, params.lightIntensity, 14, 2);
    light.position.y = HANG - 0.05;
    root.add(light);
    return { root, light, wax, posUniform: idx === 0 ? uPosA : uPosB, phase: spec.phase };
  });

  // the vintage lantern body, shared geometry, one clone per lantern; its
  // baked-in opaque "glass" part is hidden — the raymarched vessel is the glass
  new GLTFLoader().load(MODEL_URL, (gltf) => {
    const src = gltf.scene;
    src.traverse((o) => {
      if ((o as THREE.Mesh).isMesh) (o as THREE.Mesh).castShadow = true;
    });
    const glass = src.getObjectByName(MODEL_GLASS);
    if (glass) glass.visible = false;
    rigs.forEach((rig, idx) => {
      const body = idx === 0 ? src : src.clone(true);
      body.scale.setScalar(MODEL_SCALE);
      body.position.y = HANG + MODEL_LIFT;
      rig.root.add(body);
    });
  });

  function attachPane(pane: Parameters<Lanterns['attachPane']>[0]) {
    const f = pane.addFolder({ title: 'Lanterns' });
    f.addBinding(params, 'scale', { label: 'scale', min: 0.5, max: 3, step: 0.05 });
    f.addBinding(params, 'heat', { label: 'coil heat', min: 0, max: 2, step: 0.05 });
    f.addBinding(params, 'buoyancy', { label: 'buoyancy', min: 0, max: 10, step: 0.1 });
    f.addBinding(params, 'goo', { label: 'gooiness', min: 0.04, max: 0.2, step: 0.005 });
    f.addBinding(params, 'glow', { label: 'glow', min: 0, max: 3, step: 0.05 });
    f.addBinding(params, 'lightIntensity', { label: 'light', min: 0, max: 20, step: 0.1 });
  }

  let elapsed = 0;
  function update(dt: number) {
    elapsed += dt;
    uTime.value = elapsed;
    uGlow.value = params.glow;
    uHeat.value = params.heat;
    uGoo.value = params.goo;
    uScale.value = params.scale;

    const simDt = Math.min(dt, 0.05);
    for (const rig of rigs) {
      rig.root.scale.setScalar(params.scale);
      rig.posUniform.value.set(
        rig.root.position.x,
        rig.root.position.y + HANG * params.scale,
        rig.root.position.z,
      );
      // flicker: two incommensurate sines so it never quite repeats
      const t = elapsed + rig.phase;
      rig.light.intensity = params.lightIntensity *
        (0.86 + 0.10 * Math.sin(t * 7.3) + 0.05 * Math.sin(t * 13.7 + 1.4));
      rig.wax.sim.step(simDt, params.heat, params.buoyancy);
      rig.wax.sim.fill(rig.wax.blobVecs);
    }
  }

  return { group, params, attachPane, update };
}
