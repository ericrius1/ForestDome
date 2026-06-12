import * as THREE from 'three/webgpu';
import * as BufferGeometryUtils from 'three/addons/utils/BufferGeometryUtils.js';
import {
  color, mix, smoothstep, positionWorld, mx_noise_float, vec3, time,
} from 'three/tsl';
import { terrainHeight } from './terrain';

// The Helix Spire: a basalt reliquary the lanterns flank. A drum of sixteen
// twisted ribs barrels around three ring floors and a roof deck, all of it
// threaded on a glowing core column that a spiral stair climbs from the
// plinth to the sky. Amber light pools at the base, cyan takes over with
// height (the two lanterns' palettes, stretched vertically), and a torus-knot
// finial spins above the crown.
//
// Everything structural is also baked — in world space — into one merged
// collision geometry for the three-mesh-bvh capsule controller: plinth,
// steps, slabs, columns, stairs, parapets, and a few invisible helical walls
// that keep the player on the stair without reading as geometry.

export interface Sculpture {
  group: THREE.Group;
  collider: THREE.BufferGeometry; // world space, merged, for MeshBVH
  radius: number;
  baseY: number;
  update: (dt: number) => void;
}

// ---- proportions (local units = metres, y=0 at plinth top) ----------------
const R_SHELL = 7.0;     // rib drum radius at base/top (bulges mid-height)
const R_FLOOR = 6.3;     // floor slab outer radius
const R_OCULUS = 3.2;    // atrium void the stair climbs inside
const FLOOR_H = 4.2;     // storey height
const LEVELS = [FLOOR_H, FLOOR_H * 2, FLOOR_H * 3]; // slab tops (deck = last)
const H_SHELL = 14;      // rib height
const SLAB_T = 0.4;
const R_PLINTH = 8.2;
const STEPS = 30;        // stair steps per storey (one full turn each)
const RAIL_GAP = 1.75;   // radians of railing left open where the stair lands
const DOOR_HALF = 0.5;   // angular half-width of each doorway clearance
const DOOR_H = 3.2;      // lintel height — nothing structural below this in a door

const tmpM = new THREE.Matrix4();
function place(g: THREE.BufferGeometry, x: number, y: number, z: number, ry = 0): THREE.BufferGeometry {
  tmpM.makeRotationY(ry).setPosition(x, y, z);
  g.applyMatrix4(tmpM);
  return g;
}

// Vertical ribbon following a helix — invisible stair guard for the collider.
function helicalWall(
  r: number, a0: number, a1: number, y0: number, y1: number, h: number, segs: number,
): THREE.BufferGeometry {
  const pos: number[] = [];
  const uv: number[] = [];
  const idx: number[] = [];
  for (let i = 0; i <= segs; i++) {
    const t = i / segs;
    const a = a0 + (a1 - a0) * t;
    const y = y0 + (y1 - y0) * t;
    const x = Math.cos(a) * r, z = Math.sin(a) * r;
    pos.push(x, y, z, x, y + h, z);
    uv.push(t, 0, t, 1);
  }
  for (let i = 0; i < segs; i++) {
    const b = i * 2;
    idx.push(b, b + 2, b + 1, b + 1, b + 2, b + 3);
  }
  const g = new THREE.BufferGeometry();
  g.setIndex(idx);
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  g.computeVertexNormals();
  return g;
}

// rib (and ring-beam) radius at height y: a gentle barrel bulge
const drumR = (y: number) => R_SHELL * (1 + 0.16 * Math.sin(Math.PI * (y / H_SHELL)));

// ExtrudeGeometry is non-indexed while the primitives are indexed; flatten
// everything so mergeGeometries gets compatible inputs.
function mergeAll(geos: THREE.BufferGeometry[]): THREE.BufferGeometry {
  return BufferGeometryUtils.mergeGeometries(
    geos.map((g) => (g.index ? g.toNonIndexed() : g)), false,
  )!;
}

export function createSculpture(center: THREE.Vector3, entryAngle: number): Sculpture {
  // foundation: plinth top must clear the highest terrain under the footprint
  let maxT = -Infinity;
  for (let i = 0; i < 48; i++) {
    const a = (i / 48) * Math.PI * 2;
    for (const r of [0, 3.5, 6.5, 9.5]) {
      maxT = Math.max(maxT, terrainHeight(center.x + Math.cos(a) * r, center.z + Math.sin(a) * r));
    }
  }
  const baseY = maxT + 0.35;

  const group = new THREE.Group();
  group.position.set(center.x, baseY, center.z);

  // four doorways at the cardinal gaps; angular distance to the nearest one
  const doorDist = (a: number) => {
    const per = Math.PI / 2;
    let d = (a - entryAngle) % per;
    if (d < 0) d += per;
    return Math.min(d, per - d);
  };

  const stoneGeos: THREE.BufferGeometry[] = [];
  const brassGeos: THREE.BufferGeometry[] = [];
  const trimGeos: THREE.BufferGeometry[] = [];
  const colliderGeos: THREE.BufferGeometry[] = [];
  const solid = (list: THREE.BufferGeometry[], g: THREE.BufferGeometry) => {
    list.push(g);
    colliderGeos.push(g);
  };

  // ---- plinth + concentric entry steps -----------------------------------
  solid(stoneGeos, place(new THREE.CylinderGeometry(R_PLINTH, R_PLINTH + 0.7, 5, 64), 0, -2.5, 0));
  for (let i = 1; i <= 4; i++) {
    const r = R_PLINTH + i * 0.6;
    solid(stoneGeos, place(new THREE.CylinderGeometry(r, r + 0.4, 5, 64), 0, -2.5 - i * 0.3, 0));
  }

  // ---- floor slabs (annulus with the atrium oculus) -----------------------
  const slabShape = new THREE.Shape();
  slabShape.absarc(0, 0, R_FLOOR, 0, Math.PI * 2, false);
  const hole = new THREE.Path();
  hole.absarc(0, 0, R_OCULUS, 0, Math.PI * 2, true);
  slabShape.holes.push(hole);
  const slabGeo = new THREE.ExtrudeGeometry(slabShape, {
    depth: SLAB_T, bevelEnabled: false, curveSegments: 48,
  });
  slabGeo.rotateX(-Math.PI / 2); // depth now spans y ∈ [0, SLAB_T]
  for (const ly of LEVELS) {
    solid(stoneGeos, slabGeo.clone().translate(0, ly - SLAB_T, 0));
  }

  // ---- columns carrying each slab -----------------------------------------
  const colH = FLOOR_H - SLAB_T;
  for (let li = 0; li < LEVELS.length; li++) {
    const base = li * FLOOR_H;
    for (let k = 0; k < 10; k++) {
      const a = li * 0.314 + (k / 10) * Math.PI * 2;
      // ground storey: leave the doorway approaches clear of columns
      if (li === 0 && doorDist(a) < DOOR_HALF) continue;
      const x = Math.cos(a) * 5.55, z = Math.sin(a) * 5.55;
      solid(stoneGeos, place(new THREE.CylinderGeometry(0.3, 0.38, colH, 12), x, base + colH / 2, z));
      stoneGeos.push(place(new THREE.CylinderGeometry(0.52, 0.34, 0.28, 12), x, base + colH - 0.14, z));
    }
  }

  // ---- the rib drum: sixteen twisted basalt ribs --------------------------
  // Ribs that would cross a doorway below the lintel start above it instead,
  // springing from over the entry arches so each gap stays walkable.
  for (let i = 0; i < 16; i++) {
    const a0 = (i / 16) * Math.PI * 2;
    let t0 = 0;
    for (let s = 0; s <= 24; s++) {
      const t = s / 24;
      if (t * H_SHELL >= DOOR_H) break;
      if (doorDist(a0 + t * 1.9) < DOOR_HALF) { t0 = DOOR_H / H_SHELL; break; }
    }
    const pts: THREE.Vector3[] = [];
    for (let s = 0; s <= 24; s++) {
      const t = t0 + (s / 24) * (1 - t0);
      const a = a0 + t * 1.9; // ~109° of twist base to crown
      const r = drumR(t * H_SHELL);
      pts.push(new THREE.Vector3(Math.cos(a) * r, t * H_SHELL, Math.sin(a) * r));
    }
    const tube = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 48, 0.3, 8);
    solid(stoneGeos, tube);
  }

  // brass ring beams lashing the ribs at each storey; the ground-level ring
  // breaks into four arcs so it doesn't bar the doorways
  for (const ly of [0.25, FLOOR_H, FLOOR_H * 2, FLOOR_H * 3, H_SHELL - 0.1]) {
    if (ly < DOOR_H) {
      const span = Math.PI / 2 - DOOR_HALF * 2;
      for (let k = 0; k < 4; k++) {
        const beam = new THREE.TorusGeometry(drumR(ly) + 0.12, 0.16, 10, 24, span);
        beam.rotateZ(-(entryAngle + (k + 1) * (Math.PI / 2) - DOOR_HALF));
        beam.rotateX(-Math.PI / 2);
        solid(brassGeos, beam.translate(0, ly, 0));
      }
      continue;
    }
    const beam = new THREE.TorusGeometry(drumR(ly) + 0.12, 0.16, 10, 96);
    beam.rotateX(Math.PI / 2);
    solid(brassGeos, beam.translate(0, ly, 0));
  }

  // ---- parapets at every slab edge + brass caps ----------------------------
  for (const ly of LEVELS) {
    solid(stoneGeos, place(
      new THREE.CylinderGeometry(6.32, 6.32, 1.05, 64, 1, true), 0, ly + 0.525, 0,
    ));
    const cap = new THREE.TorusGeometry(6.32, 0.08, 8, 80);
    cap.rotateX(Math.PI / 2);
    brassGeos.push(cap.translate(0, ly + 1.06, 0));
  }

  // ---- atrium guard rails (arc, open where the stair lands) ----------------
  for (const ly of LEVELS) {
    // invisible collision band
    colliderGeos.push(place(new THREE.CylinderGeometry(
      3.35, 3.35, 1.05, 48, 1, true,
      Math.PI / 2 - entryAngle + RAIL_GAP / 2, Math.PI * 2 - RAIL_GAP,
    ), 0, ly + 0.525, 0));
    // visible brass rails at two heights
    for (const hy of [0.55, 1.0]) {
      const rail = new THREE.TorusGeometry(3.35, 0.05, 8, 80, Math.PI * 2 - RAIL_GAP);
      rail.rotateZ(RAIL_GAP / 2 - entryAngle);
      rail.rotateX(-Math.PI / 2);
      brassGeos.push(rail.translate(0, ly + hy, 0));
    }
  }

  // ---- spiral stair around the core ----------------------------------------
  const RISE = FLOOR_H / STEPS;
  const turn = (Math.PI * 2) / STEPS;
  const stepProto = new THREE.BoxGeometry(1.95, 0.5, 0.72);
  for (let f = 0; f < 3; f++) {
    for (let k = 0; k < STEPS; k++) {
      const a = entryAngle + (f * STEPS + k + 0.5) * turn;
      const top = f * FLOOR_H + (k + 1) * RISE;
      solid(stoneGeos, place(
        stepProto.clone(), Math.cos(a) * 2.0, top - 0.25, Math.sin(a) * 2.0, -a,
      ));
    }
    // glowing handrail riding the outer edge (gaps at both landings)
    const railPts: THREE.Vector3[] = [];
    for (let s = 0; s <= 36; s++) {
      const t = 0.06 + (s / 36) * 0.86;
      const a = entryAngle + t * Math.PI * 2;
      railPts.push(new THREE.Vector3(
        Math.cos(a) * 3.06, f * FLOOR_H + t * FLOOR_H + 1.05, Math.sin(a) * 3.06,
      ));
    }
    trimGeos.push(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(railPts), 72, 0.06, 8));
    // invisible wall under the handrail so the capsule can't drift off the edge
    colliderGeos.push(helicalWall(
      3.06,
      entryAngle + 0.06 * Math.PI * 2, entryAngle + 0.92 * Math.PI * 2,
      f * FLOOR_H + 0.06 * FLOOR_H + 0.25, f * FLOOR_H + 0.92 * FLOOR_H + 0.25,
      1.15, 48,
    ));
  }

  // ---- entry arches at the cardinal gaps -----------------------------------
  // taller frames with a glowing inner band: the doorways should read as
  // invitations from across the meadow, not gaps you have to hunt for
  for (let k = 0; k < 4; k++) {
    const a = entryAngle + (k * Math.PI) / 2;
    const arch = new THREE.TorusGeometry(2.9, 0.2, 10, 36, Math.PI);
    arch.rotateY(Math.PI / 2 - a);
    solid(brassGeos, arch.translate(Math.cos(a) * 6.9, 0, Math.sin(a) * 6.9));
    const glow = new THREE.TorusGeometry(2.62, 0.06, 8, 36, Math.PI);
    glow.rotateY(Math.PI / 2 - a);
    trimGeos.push(glow.translate(Math.cos(a) * 6.9, 0, Math.sin(a) * 6.9));
  }

  // ---- glow trim: oculus floor rings ---------------------------------------
  for (const ly of [0, ...LEVELS]) {
    const ring = new THREE.TorusGeometry(ly === 0 ? 1.8 : 3.3, 0.06, 8, 80);
    ring.rotateX(Math.PI / 2);
    trimGeos.push(ring.translate(0, ly + 0.05, 0));
  }

  // ---- materials -----------------------------------------------------------
  const wp = positionWorld;
  const stoneMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.93, metalness: 0.04, side: THREE.DoubleSide,
  });
  const strata = mx_noise_float(wp.mul(vec3(0.16, 0.85, 0.16)));
  const fleck = mx_noise_float(wp.mul(3.2)).mul(0.5).add(0.5);
  stoneMat.colorNode = mix(color(0x5a564f), color(0x2c2a27), smoothstep(-0.5, 0.7, strata))
    .mul(fleck.mul(0.18).add(0.88));

  const brassMat = new THREE.MeshStandardNodeMaterial({
    color: 0x9c7a3c, metalness: 0.85, roughness: 0.35,
  });

  // amber at the base, cyan at the crown — the lanterns' two tempers
  const amber = color(0xffa14f);
  const cyan = color(0x4fd8ff);
  const hNorm = wp.y.sub(baseY).div(15).clamp(0, 1);
  const heightCol = mix(amber, cyan, hNorm);

  const trimMat = new THREE.MeshStandardNodeMaterial({ roughness: 0.4, metalness: 0.2 });
  trimMat.colorNode = heightCol.mul(0.25);
  trimMat.emissiveNode = heightCol.mul(time.mul(1.7).sin().mul(0.18).add(0.82)).mul(2.4);

  // the core: unlit, banded light crawling upward through noise
  const coreMat = new THREE.MeshBasicNodeMaterial();
  const swirl = mx_noise_float(vec3(wp.x.mul(1.1), wp.y.mul(0.5).sub(time.mul(0.45)), wp.z.mul(1.1)));
  const bands = wp.y.mul(2.0).sub(time.mul(1.6)).add(swirl.mul(6)).sin().mul(0.5).add(0.5);
  coreMat.colorNode = heightCol.mul(bands.mul(1.3).add(0.15)).add(heightCol.mul(swirl.abs().mul(0.4)));
  coreMat.fog = true;

  // ---- meshes --------------------------------------------------------------
  const stoneMesh = new THREE.Mesh(mergeAll(stoneGeos), stoneMat);
  stoneMesh.castShadow = true;
  stoneMesh.receiveShadow = true;
  group.add(stoneMesh);

  const brassMesh = new THREE.Mesh(mergeAll(brassGeos), brassMat);
  brassMesh.castShadow = true;
  group.add(brassMesh);

  group.add(new THREE.Mesh(mergeAll(trimGeos), trimMat));

  const coreGeo = place(new THREE.CylinderGeometry(0.85, 1.1, H_SHELL + 0.9, 28), 0, H_SHELL / 2 + 0.15, 0);
  colliderGeos.push(coreGeo);
  group.add(new THREE.Mesh(coreGeo, coreMat));

  const finial = new THREE.Mesh(new THREE.TorusKnotGeometry(1.5, 0.3, 100, 16, 2, 3), coreMat);
  finial.position.y = H_SHELL + 2.3;
  group.add(finial);

  // ---- light pooling out of the core ---------------------------------------
  const l1 = new THREE.PointLight(0xffb066, 36, 28, 2);
  l1.position.set(0, 2.6, 0);
  const l2 = new THREE.PointLight(0x4fd8ff, 30, 26, 2);
  l2.position.set(0, 10.8, 0);
  const l3 = new THREE.PointLight(0x9fe8ff, 22, 30, 2);
  l3.position.set(0, H_SHELL + 2.3, 0);
  group.add(l1, l2, l3);

  // ---- world-space collider -------------------------------------------------
  const collider = mergeAll(colliderGeos);
  collider.translate(center.x, baseY, center.z);

  let elapsed = 0;
  function update(dt: number) {
    elapsed += dt;
    finial.rotation.y += dt * 0.35;
    finial.rotation.x = Math.sin(elapsed * 0.4) * 0.25;
    finial.position.y = H_SHELL + 2.3 + Math.sin(elapsed * 0.7) * 0.25;
    const pulse = 0.9 + 0.1 * Math.sin(elapsed * 1.7);
    l1.intensity = 36 * pulse;
    l2.intensity = 30 * (1.05 - 0.1 * Math.sin(elapsed * 1.7));
    l3.intensity = 22 * pulse;
  }

  return { group, collider, radius: R_PLINTH + 3, baseY, update };
}
