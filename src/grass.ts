import * as THREE from 'three/webgpu';
import {
  color, float, mix, uv, vec3, time, sin,
  positionLocal, attribute, hash, instanceIndex,
} from 'three/tsl';
import { terrainHeight, fbm } from './terrain';

export function createGrass(count = 90000, radius = 42): THREE.InstancedMesh {
  const blade = new THREE.PlaneGeometry(0.05, 0.55, 1, 3);
  blade.translate(0, 0.275, 0);

  const material = new THREE.MeshStandardNodeMaterial({
    roughness: 1, metalness: 0, side: THREE.DoubleSide,
  });

  const h = hash(instanceIndex);
  const tipness = uv().y;

  const base = mix(color(0x33471a), color(0x4d6624), h);
  const tip = mix(color(0x7e9440), color(0xa3b35e), h);
  material.colorNode = mix(base, tip, tipness.pow(1.4));

  const phase = attribute<'float'>('aPhase', 'float');
  const bend = tipness.pow(2);
  const sway = sin(time.mul(1.6).add(phase)).mul(0.13)
    .add(sin(time.mul(3.8).add(phase.mul(1.7))).mul(0.045));
  material.positionNode = positionLocal.add(vec3(sway.mul(bend), bend.mul(-0.04), sway.mul(bend).mul(0.55)));

  const mesh = new THREE.InstancedMesh(blade, material, count);
  const phases = new Float32Array(count);

  const m = new THREE.Matrix4();
  const q = new THREE.Quaternion();
  const e = new THREE.Euler();
  const v = new THREE.Vector3();
  const s = new THREE.Vector3();

  let i = 0;
  let guard = 0;
  while (i < count && guard++ < count * 6) {
    const a = Math.random() * Math.PI * 2;
    const r = Math.sqrt(Math.random()) * radius;
    const x = Math.cos(a) * r, z = Math.sin(a) * r;
    const density = 1 - fbm(x * 0.05 + 13.7, z * 0.05 + 5.2) * 1.25;
    if (Math.random() > Math.max(0.06, density)) continue;
    v.set(x, terrainHeight(x, z) - 0.02, z);
    e.set((Math.random() - 0.5) * 0.25, Math.random() * Math.PI * 2, (Math.random() - 0.5) * 0.25);
    q.setFromEuler(e);
    const sc = 0.5 + Math.random() * 1.1;
    s.set(sc, sc * (0.55 + Math.random() * 1.0), sc);
    m.compose(v, q, s);
    mesh.setMatrixAt(i, m);
    phases[i] = (x + z) * 0.55 + Math.random() * 0.9;
    i++;
  }
  mesh.count = i;
  blade.setAttribute('aPhase', new THREE.InstancedBufferAttribute(phases, 1));

  mesh.receiveShadow = true;
  return mesh;
}
