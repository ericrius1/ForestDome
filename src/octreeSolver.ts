// Barnes-Hut octree solver, ported from the blog's Pyramid3DSolver. The tree
// is rebuilt from scratch on the GPU every substep (rebuilding is cheaper
// than updating), and the force kernel integrates in place — so each body
// costs a few hundred node visits instead of one read per other body.
//
// pos/vel buffers are owned by Three.js (the sprite material reads them as
// vertex data); this class only owns the tree scratch buffers.

import shader from './octree.wgsl?raw';

const WG = 256;

// Fixed-point budget for the scatter atomics: total mass * FP_SCALE must fit
// in a u32. The tweakpane caps total mass at 100k, so size for that.
const FP_SCALE = 3.6e9 / 100000;

// Memory is the 3D tax: level F costs 8^F cells, so pick depth from the body
// count (cube root) and clamp to 128^3 — ~34 MB of accumulators at the top.
export function chooseFinestLevel(count: number): number {
  const l = Math.ceil(Math.log2(Math.max(count, 2)) / 3);
  return Math.min(7, Math.max(4, l));
}

function levelOffset(l: number): number {
  return (Math.pow(8, l) - 1) / 7;
}

export interface OctreeSimParams {
  count: number;
  dt: number;
  gravity: number;
  softening: number;
  theta: number;
  damping: number;
  shellR: number;
  shellK: number;
  floorK: number;
  maxSpeed: number;
  flow: number;
  swirl: number;
  time: number;
  pointerSoftening: number;
  /** Per-slot forcers: x, y, z, gain. */
  forcers: [number, number, number, number][];
}

const PARAMS_SIZE = 64 + 3 * 16; // 16 scalars + array<vec4f, 3>

export class OctreeDomeSolver {
  readonly count: number;
  readonly finestLevel: number;
  readonly gridDim: number;

  private dev: GPUDevice;
  private simParams: GPUBuffer;
  private grid: GPUBuffer;
  private nodes: GPUBuffer;
  private bounds: GPUBuffer;

  private pClear: GPUComputePipeline;
  private pBounds: GPUComputePipeline;
  private pScatter: GPUComputePipeline;
  private pResolve: GPUComputePipeline;
  private pReduce: GPUComputePipeline[] = [];
  private pForce: GPUComputePipeline;

  private gClear: GPUBindGroup;
  private gBounds: GPUBindGroup;
  private gScatter: GPUBindGroup;
  private gResolve: GPUBindGroup;
  private gReduce: GPUBindGroup[] = [];
  private gForce: GPUBindGroup;

  constructor(dev: GPUDevice, count: number, pos: GPUBuffer, vel: GPUBuffer) {
    this.dev = dev;
    this.count = count;
    this.finestLevel = chooseFinestLevel(count);
    this.gridDim = 1 << this.finestLevel;
    const cells = this.gridDim ** 3;

    this.simParams = dev.createBuffer({ size: PARAMS_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.grid = dev.createBuffer({ size: cells * 16, usage: GPUBufferUsage.STORAGE });
    this.nodes = dev.createBuffer({ size: levelOffset(this.finestLevel + 1) * 16, usage: GPUBufferUsage.STORAGE });
    this.bounds = dev.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE });

    const module = dev.createShaderModule({ code: shader });
    const mk = (entryPoint: string, constants: Record<string, number>): GPUComputePipeline =>
      dev.createComputePipeline({ layout: 'auto', compute: { module, entryPoint, constants } });
    const c = { FINEST: this.finestLevel, DIM: this.gridDim, FP_SCALE };
    this.pClear = mk('clear_grid', { DIM: this.gridDim });
    this.pBounds = mk('reduce_bounds', {});
    this.pScatter = mk('scatter', { DIM: this.gridDim, FP_SCALE });
    this.pResolve = mk('resolve', c);
    this.pForce = mk('force', c);
    for (let l = 0; l < this.finestLevel; l++) this.pReduce.push(mk('reduce', { LEVEL: l }));

    const grp = (pipe: GPUComputePipeline, bindings: [number, GPUBuffer][]): GPUBindGroup =>
      dev.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: bindings.map(([binding, buffer]) => ({ binding, resource: { buffer } })),
      });
    this.gClear = grp(this.pClear, [[4, this.grid], [6, this.bounds]]);
    this.gBounds = grp(this.pBounds, [[0, this.simParams], [1, pos], [6, this.bounds]]);
    this.gScatter = grp(this.pScatter, [[0, this.simParams], [1, pos], [4, this.grid], [6, this.bounds]]);
    this.gResolve = grp(this.pResolve, [[4, this.grid], [5, this.nodes], [6, this.bounds]]);
    for (let l = 0; l < this.finestLevel; l++) this.gReduce.push(grp(this.pReduce[l], [[5, this.nodes]]));
    this.gForce = grp(this.pForce, [
      [0, this.simParams], [1, pos], [2, vel], [5, this.nodes], [6, this.bounds],
    ]);
  }

  writeParams(p: OctreeSimParams): void {
    const buf = new ArrayBuffer(PARAMS_SIZE);
    const dv = new DataView(buf);
    dv.setUint32(0, this.count, true);
    dv.setFloat32(4, p.dt, true);
    dv.setFloat32(8, p.gravity, true);
    dv.setFloat32(12, p.softening, true);
    dv.setFloat32(16, p.theta, true);
    dv.setFloat32(20, p.damping, true);
    dv.setFloat32(24, p.shellR, true);
    dv.setFloat32(28, p.shellK, true);
    dv.setFloat32(32, p.floorK, true);
    dv.setFloat32(36, p.maxSpeed, true);
    dv.setFloat32(40, p.flow, true);
    dv.setFloat32(44, p.swirl, true);
    dv.setFloat32(48, p.time, true);
    dv.setFloat32(52, p.pointerSoftening, true);
    for (let s = 0; s < 3; s++) {
      const f = p.forcers[s] ?? [0, 0, 0, 0];
      for (let k = 0; k < 4; k++) dv.setFloat32(64 + s * 16 + k * 4, f[k], true);
    }
    this.dev.queue.writeBuffer(this.simParams, 0, buf);
  }

  encode(pass: GPUComputePassEncoder): void {
    const cells = this.gridDim ** 3;
    const bodyWGs = Math.ceil(this.count / WG);
    pass.setPipeline(this.pClear);
    pass.setBindGroup(0, this.gClear);
    pass.dispatchWorkgroups(Math.ceil((cells * 4) / WG));
    pass.setPipeline(this.pBounds);
    pass.setBindGroup(0, this.gBounds);
    pass.dispatchWorkgroups(bodyWGs);
    pass.setPipeline(this.pScatter);
    pass.setBindGroup(0, this.gScatter);
    pass.dispatchWorkgroups(bodyWGs);
    pass.setPipeline(this.pResolve);
    pass.setBindGroup(0, this.gResolve);
    pass.dispatchWorkgroups(Math.ceil(cells / WG));
    for (let l = this.finestLevel - 1; l >= 0; l--) {
      pass.setPipeline(this.pReduce[l]);
      pass.setBindGroup(0, this.gReduce[l]);
      pass.dispatchWorkgroups(Math.max(1, Math.ceil(8 ** l / WG)));
    }
    pass.setPipeline(this.pForce);
    pass.setBindGroup(0, this.gForce);
    pass.dispatchWorkgroups(bodyWGs);
  }

  dispose(): void {
    for (const b of [this.simParams, this.grid, this.nodes, this.bounds]) b.destroy();
  }
}
