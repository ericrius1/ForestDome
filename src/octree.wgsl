// Barnes-Hut octree pyramid, ported from the blog's pyramid3d.wgsl (part two).
// Children of cell (level, ix, iy, iz) are (level+1, 2ix+dx, 2iy+dy, 2iz+dz).
// Levels are flat slabs of one buffer; level l starts at (8^l - 1) / 7.
//
// Bodies are two buffers shared with the Three.js renderer: pos = vec4(x, y,
// z, mass), vel = vec4(vx, vy, vz, 0). The force kernel only ever writes its
// own body, so it integrates in place — no ping-pong, no copy pass.
//
// Differences from the blog shader: the world is y-up (floor spring acts on
// p.y), and the force pass carries the dome's extra O(N) terms — tangential
// noise stirring, an azimuthal breeze, three pointer/controller forcers, and
// a speed clamp.

override FINEST: u32 = 6u;    // octree depth; finest grid is DIM^3
override DIM: u32 = 64u;      // 1 << FINEST
override FP_SCALE: f32 = 1.0; // fixed-point scale for the atomic accumulators
override LEVEL: u32 = 0u;     // which level a reduce pipeline writes

struct SimParams {
  count: u32,
  dt: f32,
  g: f32,
  softening: f32,
  theta: f32,
  damping: f32,
  shellR: f32,
  shellK: f32,
  floorK: f32,
  maxSpeed: f32,
  flow: f32,
  swirl: f32,
  time: f32,
  ptrSoft: f32,
  _pad0: f32,
  _pad1: f32,
  // One forcer per slot (mouse, left hand, right hand): xyz position, w gain.
  ptr: array<vec4f, 3>,
}

@group(0) @binding(0) var<uniform> P: SimParams;
@group(0) @binding(1) var<storage, read_write> pos: array<vec4f>;  // xyz + mass
@group(0) @binding(2) var<storage, read_write> vel: array<vec4f>;  // xyz + pad
// 4 words per finest cell: mass (u32), m*dx, m*dy, m*dz (i32 bit patterns).
@group(0) @binding(4) var<storage, read_write> grid: array<atomic<u32>>;
// One vec4 per octree node: com.xyz, mass.
@group(0) @binding(5) var<storage, read_write> nodes: array<vec4f>;
// World bounds as order-preserving u32 keys: minX..minZ, maxX..maxZ.
@group(0) @binding(6) var<storage, read_write> bounds: array<atomic<u32>, 8>;

fn floatToKey(v: f32) -> u32 {
  let u = bitcast<u32>(v);
  return select(u | 0x80000000u, ~u, (u >> 31u) == 1u);
}

fn keyToFloat(k: u32) -> f32 {
  if ((k >> 31u) == 1u) {
    return bitcast<f32>(k ^ 0x80000000u);
  }
  return bitcast<f32>(~k);
}

struct RootBox {
  origin: vec3f,
  size: f32,
}

fn rootBox() -> RootBox {
  let mn = vec3f(
    keyToFloat(atomicLoad(&bounds[0])),
    keyToFloat(atomicLoad(&bounds[1])),
    keyToFloat(atomicLoad(&bounds[2])),
  );
  let mx = vec3f(
    keyToFloat(atomicLoad(&bounds[4])),
    keyToFloat(atomicLoad(&bounds[5])),
    keyToFloat(atomicLoad(&bounds[6])),
  );
  let c = (mn + mx) * 0.5;
  let half = max(max(max(mx.x - mn.x, mx.y - mn.y), mx.z - mn.z) * 0.5, 1e-6) * 1.0001;
  var r: RootBox;
  r.origin = c - vec3f(half);
  r.size = half * 2.0;
  return r;
}

// Levels 0..l-1 hold (8^l - 1) / 7 nodes.
fn levelOffset(l: u32) -> u32 {
  return ((1u << (3u * l)) - 1u) / 7u;
}

@compute @workgroup_size(256)
fn clear_grid(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i < 8u) {
    atomicStore(&bounds[i], select(0u, 0xFFFFFFFFu, i < 4u));
  }
  if (i < DIM * DIM * DIM * 4u) {
    atomicStore(&grid[i], 0u);
  }
}

var<workgroup> wmin: array<vec3f, 256>;
var<workgroup> wmax: array<vec3f, 256>;

@compute @workgroup_size(256)
fn reduce_bounds(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  var lo = vec3f(3.4e38);
  var hi = vec3f(-3.4e38);
  if (gid.x < P.count) {
    lo = pos[gid.x].xyz;
    hi = lo;
  }
  wmin[lid.x] = lo;
  wmax[lid.x] = hi;
  workgroupBarrier();
  var s = 128u;
  loop {
    if (s == 0u) {
      break;
    }
    if (lid.x < s) {
      wmin[lid.x] = min(wmin[lid.x], wmin[lid.x + s]);
      wmax[lid.x] = max(wmax[lid.x], wmax[lid.x + s]);
    }
    workgroupBarrier();
    s = s >> 1u;
  }
  if (lid.x == 0u) {
    atomicMin(&bounds[0], floatToKey(wmin[0].x));
    atomicMin(&bounds[1], floatToKey(wmin[0].y));
    atomicMin(&bounds[2], floatToKey(wmin[0].z));
    atomicMax(&bounds[4], floatToKey(wmax[0].x));
    atomicMax(&bounds[5], floatToKey(wmax[0].y));
    atomicMax(&bounds[6], floatToKey(wmax[0].z));
  }
}

@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.count) {
    return;
  }
  let rb = rootBox();
  let b = pos[i];
  let m = b.w;
  let gf = (b.xyz - rb.origin) / rb.size * f32(DIM);
  let gx = min(u32(max(gf.x, 0.0)), DIM - 1u);
  let gy = min(u32(max(gf.y, 0.0)), DIM - 1u);
  let gz = min(u32(max(gf.z, 0.0)), DIM - 1u);
  let frac = gf - vec3f(f32(gx) + 0.5, f32(gy) + 0.5, f32(gz) + 0.5);
  let c = ((gz * DIM + gy) * DIM + gx) * 4u;
  atomicAdd(&grid[c], u32(round(m * FP_SCALE)));
  atomicAdd(&grid[c + 1u], bitcast<u32>(i32(round(m * frac.x * FP_SCALE))));
  atomicAdd(&grid[c + 2u], bitcast<u32>(i32(round(m * frac.y * FP_SCALE))));
  atomicAdd(&grid[c + 3u], bitcast<u32>(i32(round(m * frac.z * FP_SCALE))));
}

@compute @workgroup_size(256)
fn resolve(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= DIM * DIM * DIM) {
    return;
  }
  let out = levelOffset(FINEST) + i;
  let mU = atomicLoad(&grid[i * 4u]);
  if (mU == 0u) {
    nodes[out] = vec4f(0.0);
    return;
  }
  let m = f32(mU) / FP_SCALE;
  let sx = f32(bitcast<i32>(atomicLoad(&grid[i * 4u + 1u]))) / FP_SCALE;
  let sy = f32(bitcast<i32>(atomicLoad(&grid[i * 4u + 2u]))) / FP_SCALE;
  let sz = f32(bitcast<i32>(atomicLoad(&grid[i * 4u + 3u]))) / FP_SCALE;
  let rb = rootBox();
  let cell = rb.size / f32(DIM);
  let ix = i % DIM;
  let iy = (i / DIM) % DIM;
  let iz = i / (DIM * DIM);
  let center = rb.origin + (vec3f(f32(ix), f32(iy), f32(iz)) + 0.5) * cell;
  let com = center + vec3f(sx, sy, sz) / m * cell;
  nodes[out] = vec4f(com, m);
}

@compute @workgroup_size(256)
fn reduce(@builtin(global_invocation_id) gid: vec3u) {
  let dim = 1u << LEVEL;
  let i = gid.x;
  if (i >= dim * dim * dim) {
    return;
  }
  let ix = i % dim;
  let iy = (i / dim) % dim;
  let iz = i / (dim * dim);
  let fineOff = levelOffset(LEVEL + 1u);
  let fdim = dim * 2u;
  var m = 0.0;
  var w = vec3f(0.0);
  for (var q = 0u; q < 8u; q = q + 1u) {
    let cx = ix * 2u + (q & 1u);
    let cy = iy * 2u + ((q >> 1u) & 1u);
    let cz = iz * 2u + (q >> 2u);
    let n = nodes[fineOff + (cz * fdim + cy) * fdim + cx];
    m = m + n.w;
    w = w + n.xyz * n.w;
  }
  var out = vec4f(0.0);
  if (m > 0.0) {
    out = vec4f(w / m, m);
  }
  nodes[levelOffset(LEVEL) + i] = out;
}

// Cheap 3D value noise in roughly [-1, 1] — stands in for the TSL version's
// mx_noise_float as the stirring field. Quality barely matters here; the
// field just has to vary smoothly in space and time.
fn hash3(p: vec3f) -> f32 {
  var q = fract(p * vec3f(0.1031, 0.1030, 0.0973));
  q = q + dot(q, q.yxz + 33.33);
  return fract((q.x + q.y) * q.z);
}

fn noise3(p: vec3f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let n00 = mix(hash3(i + vec3f(0.0, 0.0, 0.0)), hash3(i + vec3f(1.0, 0.0, 0.0)), u.x);
  let n10 = mix(hash3(i + vec3f(0.0, 1.0, 0.0)), hash3(i + vec3f(1.0, 1.0, 0.0)), u.x);
  let n01 = mix(hash3(i + vec3f(0.0, 0.0, 1.0)), hash3(i + vec3f(1.0, 0.0, 1.0)), u.x);
  let n11 = mix(hash3(i + vec3f(0.0, 1.0, 1.0)), hash3(i + vec3f(1.0, 1.0, 1.0)), u.x);
  return mix(mix(n00, n10, u.y), mix(n01, n11, u.y), u.z) * 2.0 - 1.0;
}

@compute @workgroup_size(256)
fn force(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.count) {
    return;
  }
  let b = pos[i];
  let p = b.xyz;
  let myMass = b.w;
  var v = vel[i].xyz;

  let rb = rootBox();
  let gf = (p - rb.origin) / rb.size * f32(DIM);
  let myIx = min(u32(max(gf.x, 0.0)), DIM - 1u);
  let myIy = min(u32(max(gf.y, 0.0)), DIM - 1u);
  let myIz = min(u32(max(gf.z, 0.0)), DIM - 1u);

  let eps2 = P.softening * P.softening;
  let theta2 = P.theta * P.theta;
  var acc = vec3f(0.0);

  // Entries pack level (4 bits) | iz | iy | ix (9 bits each). An octree pop
  // pushes 8, so the stack peaks at 7*FINEST+8; 64 covers FINEST = 7.
  var stack: array<u32, 64>;
  var sp: i32 = 1;
  stack[0] = 0u;

  loop {
    if (sp == 0) {
      break;
    }
    sp = sp - 1;
    let e = stack[sp];
    let lvl = e >> 28u;
    let ix = e & 0x1FFu;
    let iy = (e >> 9u) & 0x1FFu;
    let iz = (e >> 18u) & 0x1FFu;
    let ldim = 1u << lvl;
    let n = nodes[levelOffset(lvl) + (iz * ldim + iy) * ldim + ix];
    if (n.w <= 0.0) {
      continue;
    }
    let d0 = n.xyz - p;
    let r2 = dot(d0, d0) + eps2;
    let w = rb.size / f32(ldim);
    if (lvl == FINEST) {
      // The finest cell's center of mass includes me; subtract myself out so
      // a body never gravitates toward itself.
      var m = n.w;
      var com = n.xyz;
      if (ix == myIx && iy == myIy && iz == myIz) {
        m = m - myMass;
        if (m <= 1e-9) {
          continue;
        }
        com = (n.xyz * n.w - p * myMass) / m;
      }
      let d = com - p;
      let rr = dot(d, d) + eps2;
      acc = acc + d * (P.g * m / (rr * sqrt(rr)));
    } else if (w * w < theta2 * r2) {
      acc = acc + d0 * (P.g * n.w / (r2 * sqrt(r2)));
    } else if (sp <= 55) {
      let cl = lvl + 1u;
      let bx = ix * 2u;
      let by = iy * 2u;
      let bz = iz * 2u;
      for (var q = 0u; q < 8u; q = q + 1u) {
        stack[sp] = (cl << 28u)
          | ((bz + (q >> 2u)) << 18u)
          | ((by + ((q >> 1u) & 1u)) << 9u)
          | (bx + (q & 1u));
        sp = sp + 1;
      }
    }
  }

  // The dome: a radial spring holds each body near the shell, and a stiffer
  // floor spring keeps it in the upper (y > 0) hemisphere.
  let r = length(p);
  let rhat = p / max(r, 1e-6);
  if (P.shellK > 0.0) {
    acc = acc - rhat * (r - P.shellR) * P.shellK;
    if (p.y < 0.0) {
      acc.y = acc.y - p.y * P.shellK * P.floorK;
    }
  }

  // Gentle stirring: a noise vector field with its radial component removed
  // pushes particles along the shell surface; an azimuthal breeze (zero at
  // the pole) feeds back just enough energy to offset damping.
  if (P.flow > 0.0) {
    let fp = p * 2.2;
    let ft = P.time * 0.12;
    let fv = vec3f(
      noise3(fp + vec3f(ft, 0.0, 17.3)),
      noise3(fp + vec3f(31.7, ft, 5.1)),
      noise3(fp + vec3f(9.2, 23.4, ft)),
    );
    acc = acc + (fv - rhat * dot(fv, rhat)) * P.flow;
  }
  let az = vec3f(-p.z, 0.0, p.x) / max(r, 1e-6);
  acc = acc + az * P.swirl;

  // Pointer / VR controller penalty forces (one slot per hand + mouse).
  for (var s = 0u; s < 3u; s = s + 1u) {
    let f = P.ptr[s];
    let pd = f.xyz - p;
    let pr2 = dot(pd, pd) + P.ptrSoft * P.ptrSoft;
    acc = acc + pd * (f.w / (pr2 * sqrt(pr2)));
  }

  v = (v + acc * P.dt) * P.damping;
  let sp2 = length(v);
  if (sp2 > P.maxSpeed) {
    v = v * (P.maxSpeed / sp2);
  }
  pos[i] = vec4f(p + v * P.dt, myMass);
  vel[i] = vec4f(v, 0.0);
}
