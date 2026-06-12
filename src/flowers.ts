import * as THREE from 'three/webgpu';
import {
  color, float, mix, smoothstep, uv, atan, cos, length, step,
} from 'three/tsl';
import { terrainHeight } from './terrain';

export function createFlowers(count = 450, radius = 38): THREE.InstancedMesh {
  const geo = new THREE.PlaneGeometry(0.16, 0.16);

  const material = new THREE.MeshStandardNodeMaterial({
    roughness: 1, metalness: 0, side: THREE.DoubleSide,
  });

  const p = uv().sub(0.5);
  const r = length(p);
  const ang = atan(p.y, p.x);
  const petalRadius = cos(ang.mul(5)).mul(0.16).add(0.30);
  const mask = step(r, petalRadius);

  material.opacityNode = mask;
  material.alphaTestNode = float(0.5);
  material.colorNode = mix(color(0xe9c64f), color(0xf7f4ea), smoothstep(0.04, 0.10, r));
  material.emissiveNode = mix(color(0x3a3110), color(0x35332c), smoothstep(0.04, 0.10, r));

  const mesh = new THREE.InstancedMesh(geo, material, count);

  const m = new THREE.Matrix4();
  const q = new THREE.Quaternion();
  const e = new THREE.Euler();
  const v = new THREE.Vector3();
  const s = new THREE.Vector3();

  for (let i = 0; i < count; i++) {
    const a = Math.random() * Math.PI * 2;
    const rr = Math.sqrt(Math.random()) * radius;
    const x = Math.cos(a) * rr, z = Math.sin(a) * rr;
    v.set(x, terrainHeight(x, z) + 0.10 + Math.random() * 0.1, z);
    e.set(-Math.PI / 2 + (Math.random() - 0.5) * 0.7, 0, Math.random() * Math.PI * 2);
    q.setFromEuler(e);
    const sc = 0.7 + Math.random() * 0.8;
    s.set(sc, sc, sc);
    m.compose(v, q, s);
    mesh.setMatrixAt(i, m);
  }

  mesh.receiveShadow = true;
  return mesh;
}
