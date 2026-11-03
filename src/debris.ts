import * as THREE from "three/webgpu";

// Bits thrown out of a blast: a fixed pool of instances flung in a cone
// along a direction, falling under gravity and shrinking away. Each one is
// stretched along its motion, so fast bits streak

export type DebrisStyle = {
  pool: number;
  perBurst: number;
  life: [number, number]; // seconds, random between
  speed: [number, number]; // metres per second, random between
  carry: number; // the burst's direction in a bit's heading
  fan: number; // the random part, which sets how wide the cone is
  lift: number; // extra upwards in the heading
  gravity: number; // metres per second squared
  drag: number; // fraction of speed lost per second
  width: number; // metres
  // Height at which a falling bit reports landing, before it reaches the
  // ground: the grass tops it falls into
  landAt?: number;
  // Where bits start: a column around the burst point reaching this far
  // below it, instead of the point itself
  body?: { radius: number; below: number };
  variety?: number; // each bit's size varies by up to this fraction either way
  stretch: number; // length per metre per second of speed, on top of width
  // How size falls off over life: 1 shrinks evenly, lower holds size longer
  shrink: number;
  // How many times its size it swells by over its life, on top
  grow?: number;
};

type Bit = {
  position: THREE.Vector3;
  velocity: THREE.Vector3;
  age: number;
  life: number;
  size: number;
  landed: boolean;
};

const zAxis = new THREE.Vector3(0, 0, 1);
const matrix = new THREE.Matrix4();
const rotation = new THREE.Quaternion();
const scale = new THREE.Vector3();
const heading = new THREE.Vector3();
const hidden = new THREE.Matrix4().makeScale(0, 0, 0);

export class Debris {
  readonly mesh: THREE.InstancedMesh;
  private readonly bits: Bit[] = [];
  private next = 0;

  constructor(
    private readonly style: DebrisStyle,
    geometry: THREE.BufferGeometry,
    material: THREE.Material,
    // Told where each bit hits the ground, and how wide it was
    private readonly onLand?: (position: THREE.Vector3, width: number) => void
  ) {
    this.mesh = new THREE.InstancedMesh(geometry, material, style.pool);
    this.mesh.frustumCulled = false;
    for (let i = 0; i < style.pool; i++) {
      this.mesh.setMatrixAt(i, hidden);
      this.bits.push({ position: new THREE.Vector3(), velocity: new THREE.Vector3(), age: 1, life: 0, size: 1, landed: true });
    }
  }

  // Throw a burst from a point on along a level direction
  burst(from: THREE.Vector3, direction: THREE.Vector3) {
    const { style } = this;
    for (let i = 0; i < style.perBurst; i++) {
      const bit = this.bits[this.next];
      this.next = (this.next + 1) % style.pool;
      bit.position.copy(from);
      if (style.body) {
        const angle = Math.random() * Math.PI * 2;
        const reach = Math.sqrt(Math.random()) * style.body.radius;
        bit.position.x += Math.cos(angle) * reach;
        bit.position.z += Math.sin(angle) * reach;
        bit.position.y -= Math.random() * style.body.below;
      }
      bit.velocity
        .randomDirection()
        .multiplyScalar(style.fan)
        .addScaledVector(direction, style.carry);
      bit.velocity
        .setY(Math.abs(bit.velocity.y) + style.lift)
        .normalize()
        .multiplyScalar(THREE.MathUtils.randFloat(...style.speed));
      bit.age = 0;
      bit.landed = false;
      bit.life = THREE.MathUtils.randFloat(...style.life);
      bit.size = 1 + THREE.MathUtils.randFloatSpread(2 * (style.variety ?? 0));
    }
  }

  update(delta: number) {
    const { style } = this;
    const keep = Math.max(1 - style.drag * delta, 0);
    for (let i = 0; i < style.pool; i++) {
      const bit = this.bits[i];
      if (bit.age >= bit.life) continue;
      bit.age += delta;
      // Gone once spent, or on reaching the ground
      if (!bit.landed && bit.velocity.y < 0 && bit.position.y < (style.landAt ?? 0)) {
        bit.landed = true;
        this.onLand?.(bit.position, style.width * bit.size);
      }
      if (bit.age >= bit.life || bit.position.y < 0) {
        bit.age = bit.life;
        this.mesh.setMatrixAt(i, hidden);
        continue;
      }
      bit.velocity.multiplyScalar(keep);
      bit.velocity.y -= style.gravity * delta;
      bit.position.addScaledVector(bit.velocity, delta);
      const speed = Math.max(bit.velocity.length(), 1e-3);
      const t = bit.age / bit.life;
      const size = bit.size * (1 - t) ** style.shrink * (1 + (style.grow ?? 0) * t);
      rotation.setFromUnitVectors(zAxis, heading.copy(bit.velocity).divideScalar(speed));
      scale.set(style.width * size, style.width * size, (style.width + speed * style.stretch) * size);
      this.mesh.setMatrixAt(i, matrix.compose(bit.position, rotation, scale));
    }
    this.mesh.instanceMatrix.needsUpdate = true;
  }
}
