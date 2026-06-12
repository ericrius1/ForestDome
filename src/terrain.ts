import * as THREE from 'three/webgpu';
import {
  color, mix, max, smoothstep, positionWorld, normalWorld, mx_noise_float, vec3,
} from 'three/tsl';

function hash2(ix: number, iz: number): number {
  const h = Math.sin(ix * 127.1 + iz * 311.7) * 43758.5453123;
  return h - Math.floor(h);
}

function valueNoise(x: number, z: number): number {
  const ix = Math.floor(x), iz = Math.floor(z);
  const fx = x - ix, fz = z - iz;
  const sx = fx * fx * (3 - 2 * fx), sz = fz * fz * (3 - 2 * fz);
  const a = hash2(ix, iz), b = hash2(ix + 1, iz);
  const c = hash2(ix, iz + 1), d = hash2(ix + 1, iz + 1);
  return a + (b - a) * sx + (c - a) * sz + (a - b - c + d) * sx * sz;
}

export function fbm(x: number, z: number, octaves = 4): number {
  let v = 0, amp = 0.5, f = 1;
  for (let i = 0; i < octaves; i++) {
    v += amp * valueNoise(x * f, z * f);
    amp *= 0.5;
    f *= 2.03;
  }
  return v;
}

export function terrainHeight(x: number, z: number): number {
  return fbm(x * 0.025, z * 0.025) * 11 - 5.5
       + fbm(x * 0.11 + 7.3, z * 0.11 + 2.1) * 1.5;
}

export function createTerrain(): THREE.Mesh {
  const size = 170;
  const segs = 220;
  const geometry = new THREE.PlaneGeometry(size, size, segs, segs);
  geometry.rotateX(-Math.PI / 2);

  const pos = geometry.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), z = pos.getZ(i);
    pos.setY(i, terrainHeight(x, z));
  }
  geometry.computeVertexNormals();

  const material = new THREE.MeshStandardNodeMaterial({ roughness: 1, metalness: 0 });

  const wp = positionWorld;
  const patch = mx_noise_float(wp.xz.mul(0.045)).add(mx_noise_float(wp.xz.mul(0.18)).mul(0.45));
  let dirtMask = smoothstep(0.25, 0.62, patch);
  const slopeMask = smoothstep(0.92, 0.78, normalWorld.y);
  dirtMask = max(dirtMask, slopeMask);

  const grassCol = mix(color(0x3c5423), color(0x547036), mx_noise_float(wp.xz.mul(0.5)).mul(0.5).add(0.5));
  const dirtCol = mix(color(0x4f3c2c), color(0x6b5440), mx_noise_float(wp.xz.mul(0.9)).mul(0.5).add(0.5));

  const detail = mx_noise_float(wp.xz.mul(1.6)).mul(0.10).add(0.95);
  material.colorNode = mix(grassCol, dirtCol, dirtMask).mul(detail);

  const mesh = new THREE.Mesh(geometry, material);
  mesh.receiveShadow = true;
  return mesh;
}
