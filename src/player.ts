import * as THREE from 'three/webgpu';
import { MeshBVH } from 'three-mesh-bvh';
import { terrainHeight } from './terrain';

// Capsule-vs-BVH character collision, after three-mesh-bvh's
// characterMovement example. The collider geometry is baked in world space,
// so the capsule segment needs no space conversion: push the segment out of
// every intersecting triangle until it's clean (or three passes, whichever
// comes first). The analytic terrain stays a height function rather than
// triangles — it's the floor of last resort everywhere.

export interface PlayerPhysics {
  radius: number;
  height: number;
  // Mutates `feet` (the capsule's lowest point) to a non-penetrating
  // position. Returns corrected vertical velocity and ground contact.
  resolve: (feet: THREE.Vector3, velY: number) => { velY: number; grounded: boolean };
}

export function createPlayerPhysics(
  collisionGeo: THREE.BufferGeometry,
  radius = 0.35,
  height = 1.75,
): PlayerPhysics {
  const bvh = new MeshBVH(collisionGeo);
  const segment = new THREE.Line3();
  const aabb = new THREE.Box3();
  const triPoint = new THREE.Vector3();
  const capPoint = new THREE.Vector3();
  const push = new THREE.Vector3();

  function resolve(feet: THREE.Vector3, velY: number) {
    let grounded = false;

    const ty = terrainHeight(feet.x, feet.z);
    if (feet.y <= ty) {
      feet.y = ty;
      grounded = true;
      if (velY < 0) velY = 0;
    }

    segment.start.set(feet.x, feet.y + radius, feet.z);
    segment.end.set(feet.x, feet.y + height - radius, feet.z);

    let pushedUp = false;
    let pushedDown = false;
    for (let pass = 0; pass < 3; pass++) {
      let any = false;
      aabb.makeEmpty();
      aabb.expandByPoint(segment.start);
      aabb.expandByPoint(segment.end);
      aabb.min.addScalar(-radius);
      aabb.max.addScalar(radius);
      bvh.shapecast({
        intersectsBounds: (box) => box.intersectsBox(aabb),
        intersectsTriangle: (tri) => {
          const dist = tri.closestPointToSegment(segment, triPoint, capPoint);
          if (dist < radius - 1e-5) {
            push.copy(capPoint).sub(triPoint);
            if (push.lengthSq() < 1e-12) push.set(0, 1, 0);
            else push.normalize();
            const depth = radius - dist;
            segment.start.addScaledVector(push, depth);
            segment.end.addScaledVector(push, depth);
            if (push.y > 0.5) pushedUp = true;
            else if (push.y < -0.5) pushedDown = true;
            any = true;
          }
        },
      });
      if (!any) break;
    }

    feet.set(segment.start.x, segment.start.y - radius, segment.start.z);
    if (pushedUp) {
      grounded = true;
      if (velY < 0) velY = 0;
    }
    if (pushedDown && velY > 0) velY = 0;
    return { velY, grounded };
  }

  return { radius, height, resolve };
}
