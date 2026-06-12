import * as THREE from 'three/webgpu';
import * as BufferGeometryUtils from 'three/addons/utils/BufferGeometryUtils.js';
import {
  color, float, mix, max, smoothstep, step, uv, vec3, time, sin,
  positionLocal, positionWorld, attribute, hash, instanceIndex, mx_noise_float,
} from 'three/tsl';
import { terrainHeight } from './terrain';
import { DOME, CLEARING_R } from './nbodyDome';

type Rng = () => number;

interface LeafCluster {
  center: THREE.Vector3;
  r: number;
}

function makeRng(seed: number): Rng {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

const UP = new THREE.Vector3(0, 1, 0);

function limbGeometry(
  base: THREE.Vector3, dir: THREE.Vector3, length: number,
  rBase: number, rTip: number, bend: number, rng: Rng,
): THREE.BufferGeometry {
  const geo = new THREE.CylinderGeometry(rTip, rBase, length, 7, 5);
  geo.translate(0, length / 2, 0);

  const pos = geo.attributes.position;
  const phase = rng() * Math.PI * 2;
  for (let i = 0; i < pos.count; i++) {
    const t = pos.getY(i) / length;
    const bx = Math.sin(t * 1.7 + phase) * bend * t;
    const bz = Math.cos(t * 1.3 + phase) * bend * t * 0.7;
    pos.setX(i, pos.getX(i) + bx);
    pos.setZ(i, pos.getZ(i) + bz);
  }

  const quat = new THREE.Quaternion().setFromUnitVectors(UP, dir.clone().normalize());
  const m = new THREE.Matrix4().compose(base, quat, new THREE.Vector3(1, 1, 1));
  geo.applyMatrix4(m);
  return geo;
}

function buildOak(
  origin: THREE.Vector3, scale: number, rng: Rng,
  barkGeos: THREE.BufferGeometry[], clusters: LeafCluster[],
): void {
  const height = (5.5 + rng() * 3.5) * scale;
  const rBase = (0.32 + rng() * 0.22) * scale;
  const lean = new THREE.Vector3((rng() - 0.5) * 0.35, 1, (rng() - 0.5) * 0.35).normalize();

  barkGeos.push(limbGeometry(origin, lean, height, rBase, rBase * 0.45, 0.5 * scale, rng));
  const top = origin.clone().addScaledVector(lean, height);

  const branchCount = 3 + Math.floor(rng() * 3);
  for (let i = 0; i < branchCount; i++) {
    const t = 0.55 + rng() * 0.38;
    const start = origin.clone().addScaledVector(lean, height * t);
    const ang = rng() * Math.PI * 2;
    const upness = 0.35 + rng() * 0.55;
    const dir = new THREE.Vector3(Math.cos(ang), upness, Math.sin(ang)).normalize();
    const len = (2.0 + rng() * 2.4) * scale * (1.15 - t * 0.4);
    barkGeos.push(limbGeometry(start, dir, len, rBase * 0.42 * (1.1 - t * 0.5), rBase * 0.12, 0.6 * scale, rng));

    clusters.push({ center: start.clone().addScaledVector(dir, len * 0.95), r: (1.5 + rng() * 1.1) * scale });
  }
  clusters.push({ center: top, r: (1.9 + rng() * 1.2) * scale });
  clusters.push({
    center: top.clone().add(new THREE.Vector3((rng() - 0.5) * 2.4 * scale, -0.6 * scale, (rng() - 0.5) * 2.4 * scale)),
    r: (1.6 + rng() * 1.0) * scale,
  });
}

function buildBirch(
  origin: THREE.Vector3, scale: number, rng: Rng,
  barkGeos: THREE.BufferGeometry[], clusters: LeafCluster[],
): void {
  const height = (7 + rng() * 3) * scale;
  const rBase = (0.13 + rng() * 0.06) * scale;
  const lean = new THREE.Vector3((rng() - 0.5) * 0.2, 1, (rng() - 0.5) * 0.2).normalize();

  barkGeos.push(limbGeometry(origin, lean, height, rBase, rBase * 0.35, 0.35 * scale, rng));
  const top = origin.clone().addScaledVector(lean, height);

  clusters.push({ center: top, r: (1.1 + rng() * 0.7) * scale });
  clusters.push({
    center: top.clone().add(new THREE.Vector3((rng() - 0.5) * 1.6, -1.1 * scale, (rng() - 0.5) * 1.6)),
    r: (0.9 + rng() * 0.6) * scale,
  });
}

function oakBarkMaterial(): THREE.MeshStandardNodeMaterial {
  const mat = new THREE.MeshStandardNodeMaterial({ roughness: 1, metalness: 0 });
  const n = mx_noise_float(positionWorld.mul(vec3(2.4, 0.55, 2.4))).mul(0.5).add(0.5);
  const fine = mx_noise_float(positionWorld.mul(vec3(9, 2.5, 9))).mul(0.12).add(0.94);
  mat.colorNode = mix(color(0x46321f), color(0x6b5138), n).mul(fine);
  return mat;
}

function birchBarkMaterial(): THREE.MeshStandardNodeMaterial {
  const mat = new THREE.MeshStandardNodeMaterial({ roughness: 0.9, metalness: 0 });
  const bands = smoothstep(0.30, 0.42, mx_noise_float(positionWorld.mul(vec3(1.6, 7.0, 1.6))));
  mat.colorNode = mix(color(0xe6e3d6), color(0x2c2823), bands);
  return mat;
}

function leafMaterial(): THREE.MeshStandardNodeMaterial {
  const mat = new THREE.MeshStandardNodeMaterial({
    roughness: 1, metalness: 0, side: THREE.DoubleSide,
  });

  const p = uv().sub(0.5);
  // TSL node types widen through max(); keep untyped for the rosette loop.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let mask: any = float(0);
  const N = 7;
  for (let i = 0; i < N; i++) {
    const ang = (i / N) * Math.PI * 2 + 0.4;
    const ca = Math.cos(ang), sa = Math.sin(ang);
    const cx = ca * 0.20, cy = sa * 0.20;
    const dx = p.x.sub(cx), dy = p.y.sub(cy);
    const lx = dx.mul(ca).add(dy.mul(sa));
    const ly = dx.mul(-sa).add(dy.mul(ca));
    const d = lx.div(0.34).pow(2).add(ly.div(0.10).pow(2));
    mask = max(mask, float(1).sub(d));
  }
  mask = max(mask, float(1).sub(p.length().div(0.16).pow(2)));

  mat.opacityNode = step(0.02, mask);
  mat.alphaTestNode = float(0.5);

  const h = hash(instanceIndex);
  const clusterTint = mx_noise_float(positionWorld.mul(0.35)).mul(0.5).add(0.5);
  const leafCol = mix(color(0x243d14), color(0x6a8a36), h.mul(0.5).add(clusterTint.mul(0.5)))
    .mul(mix(float(0.7), float(1.18), smoothstep(0.0, 0.45, p.length())));
  mat.colorNode = leafCol;

  const phase = attribute<'float'>('aPhase', 'float');
  const sway = sin(time.mul(1.3).add(phase)).mul(0.06)
    .add(sin(time.mul(2.9).add(phase.mul(1.7))).mul(0.03));
  mat.positionNode = positionLocal.add(vec3(sway, sway.mul(0.5), 0));

  return mat;
}

export function createForest(): THREE.Group {
  const group = new THREE.Group();
  const rng = makeRng(1337);

  const oakBark: THREE.BufferGeometry[] = [];
  const birchBark: THREE.BufferGeometry[] = [];
  const clusters: LeafCluster[] = [];

  const placed: { x: number; z: number }[] = [];
  const tryPlace = (minR: number, maxR: number, minDist: number): THREE.Vector3 | null => {
    for (let attempt = 0; attempt < 40; attempt++) {
      const a = rng() * Math.PI * 2;
      const r = minR + rng() * (maxR - minR);
      const x = Math.cos(a) * r, z = Math.sin(a) * r;
      if (z > 12 && Math.abs(x - 3) < 8) continue;
      if ((x - DOME.x) ** 2 + (z - DOME.z) ** 2 < CLEARING_R ** 2) continue;
      if (placed.every((p) => (p.x - x) ** 2 + (p.z - z) ** 2 > minDist * minDist)) {
        placed.push({ x, z });
        return new THREE.Vector3(x, terrainHeight(x, z) - 0.25, z);
      }
    }
    return null;
  };

  buildOak(new THREE.Vector3(16, terrainHeight(16, 6) - 0.3, 6), 1.5, rng, oakBark, clusters);
  placed.push({ x: 16, z: 6 });

  for (let i = 0; i < 17; i++) {
    const o = tryPlace(8, 55, 6.5);
    if (o) buildOak(o, 0.8 + rng() * 0.8, rng, oakBark, clusters);
  }
  for (let i = 0; i < 6; i++) {
    const o = tryPlace(7, 45, 5);
    if (o) buildBirch(o, 0.9 + rng() * 0.5, rng, birchBark, clusters);
  }

  const oakMesh = new THREE.Mesh(BufferGeometryUtils.mergeGeometries(oakBark), oakBarkMaterial());
  oakMesh.castShadow = true;
  oakMesh.receiveShadow = true;
  group.add(oakMesh);

  if (birchBark.length) {
    const birchMesh = new THREE.Mesh(BufferGeometryUtils.mergeGeometries(birchBark), birchBarkMaterial());
    birchMesh.castShadow = true;
    birchMesh.receiveShadow = true;
    group.add(birchMesh);
  }

  const cardsPerCluster = 38;
  const count = clusters.length * cardsPerCluster;
  const cardGeo = new THREE.PlaneGeometry(1, 1);
  const phases = new Float32Array(count);
  const leaves = new THREE.InstancedMesh(cardGeo, leafMaterial(), count);

  const m = new THREE.Matrix4();
  const q = new THREE.Quaternion();
  const e = new THREE.Euler();
  const v = new THREE.Vector3();
  const s = new THREE.Vector3();

  let idx = 0;
  for (const c of clusters) {
    for (let i = 0; i < cardsPerCluster; i++) {
      const a = rng() * Math.PI * 2;
      const u = rng() * 2 - 1;
      const rr = Math.cbrt(rng());
      const sq = Math.sqrt(1 - u * u);
      v.set(
        c.center.x + Math.cos(a) * sq * rr * c.r * 1.15,
        c.center.y + u * rr * c.r * 0.75,
        c.center.z + Math.sin(a) * sq * rr * c.r * 1.15,
      );
      e.set((rng() - 0.5) * 1.4, rng() * Math.PI * 2, (rng() - 0.5) * 0.9);
      q.setFromEuler(e);
      const sc = (0.8 + rng() * 0.8) * Math.max(0.8, c.r * 0.6);
      s.set(sc, sc, sc);
      m.compose(v, q, s);
      leaves.setMatrixAt(idx, m);
      phases[idx] = (v.x + v.z) * 0.45 + rng() * 0.8;
      idx++;
    }
  }
  cardGeo.setAttribute('aPhase', new THREE.InstancedBufferAttribute(phases, 1));
  leaves.castShadow = true;
  leaves.receiveShadow = true;
  group.add(leaves);

  return group;
}
