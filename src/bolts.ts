import * as THREE from "three/webgpu";
import {
  Discard,
  Fn,
  If,
  length,
  max,
  mrt,
  output,
  vec4,
  normalLocal,
  normalView,
  positionLocal,
  pow,
  saturate,
  sin,
  uniform,
  varying,
  vec3,
} from "three/tsl";
import { Debris, type DebrisStyle } from "./debris";
import type { Stains } from "./stains";

// Red laser bolts, shared by everyone who shoots. Each flies level from
// where it was fired and ends on the first target of another team it
// passes through, or after BOLT_LIFE. A hit on something that explodes
// flashes its light, and the bolt splats against it, with sparks and blood
// blown on the way the bolt was going

const BOLT_SPEED = 70; // metres per second
const BOLT_LIFE = 0.7; // seconds
// Each bolt leaves the muzzle as a short fat orb and stretches out into a
// bolt over this long, its tail held at the muzzle until then
const BOLT_GROW = 0.09; // seconds
const ORB_WIDTH = 11; // times the bolt's radius
const ORB_LENGTH = 0.14; // times the bolt's length
const BOLT_LENGTH = 4.4;
const BOLT_RADIUS = 0.035;

// Enough for every bolt that can be alive at once: the player's at full
// rate and the enemies'. Each has its own red point light, hidden while
// idle: clustered lighting doesn't recompile materials when point lights
// come and go, but every light it holds costs on every lit pixel near it
const POOL_SIZE = 14;
const LIGHT_INTENSITY = 32;
const LIGHT_COLOR = "#ff1006";
const LIGHT_RANGE = 10;
// The light flashes this many times brighter as the orb leaves the muzzle,
// dying down as it stretches into a bolt
const MUZZLE_FLASH = 4;

// The burst on an exploding hit: the light flashes and dies over
// BURST_TIME. The bolt keeps flowing into the impact, slowed, and splats:
// whatever reaches the impact squashes outwards into a cone splashing back
// the way it came, and is gone once it has spread SPLAT_REACH
const BURST_TIME = 0.25; // seconds
const BURST_LIGHT = 380; // light intensity at the flash
const BURST_LIGHT_RANGE = 16;
const SPLAT_SPEED = 40; // metres per second, flowing into the impact
const SPLAT_SPREAD = 0.4; // metres outwards per metre that has flowed in
const SPLAT_BACK = 0.12; // metres back per metre that has flowed in
const SPLAT_REACH = 0.6; // metres flowed in before it's gone
// Where the bolt's mesh is cut while flying: nowhere
const UNCUT = 1e6;

// Sparks: hot streaks blown on in a cone along the bolt's way, slowing and
// falling
const SPARKS: DebrisStyle = {
  pool: 330,
  perBurst: 70,
  life: [0.3, 0.65],
  speed: [11, 23],
  carry: 2.6,
  fan: 1.9,
  lift: 0.1,
  gravity: 22,
  drag: 1.6,
  width: 0.18,
  stretch: 0.065,
  shrink: 1,
  // From all over the splat rather than one point
  body: { radius: 0.3, below: 0.15 },
};

// Blood: dark, chunky blobs thrown on the same way, heavier and slower,
// that fall to the grass
const BLOOD: DebrisStyle = {
  pool: 80,
  perBurst: 12,
  life: [0.8, 1.6],
  speed: [5.75, 15],
  carry: 2.2,
  fan: 0.7,
  lift: 0.5,
  gravity: 22,
  drag: 0.8,
  width: 0.37,
  // Dents the grass as it falls into the blade tops
  landAt: 1.3,
  // Thrown from all of the body that was hit, down to its knees
  body: { radius: 0.35, below: 0.9 },
  variety: 0.5,
  stretch: 0.01,
  shrink: 0.3,
};

// A blood blob: a rough ball pointing along +z, the way it flies, drawn out
// behind into a thinning tail
function teardrop() {
  const geometry = new THREE.IcosahedronGeometry(0.5, 1);
  const position = geometry.attributes.position;
  for (let i = 0; i < position.count; i++) {
    const z = position.getZ(i);
    if (z >= 0) continue;
    const back = -z / 0.5;
    const thin = 1 - 0.75 * back;
    position.setXYZ(i, position.getX(i) * thin, position.getY(i) * thin, z * 2.2);
  }
  geometry.computeVertexNormals();
  return geometry;
}

// Opacity at the middle of a blood mist puff
const MIST_OPACITY = 0.26;

// How far blood blobs' surfaces bulge in and out as they fly, in metres
const BLOOD_WOBBLE = 0.07;

// A blob's dent in the grass, as a multiple of its width
const DENT_SIZE = 2.2;

// Blood mist: a faint red cloud puffed out of the body, swelling as it
// rolls out the bolt's way over the splatter as that spreads, slowing as
// it goes, and thins out
const MIST: DebrisStyle = {
  pool: 84,
  perBurst: 14,
  life: [0.9, 1.4],
  speed: [8, 16],
  carry: 3,
  fan: 0.7,
  lift: 0.15,
  gravity: 1,
  drag: 2,
  // Starts out big and peaks early, so it bursts in rather than swells
  width: 1.6,
  variety: 0.4,
  stretch: 0,
  shrink: 1,
  grow: 2.5,
  body: { radius: 0.3, below: 0.7 },
};

export type Team = "player" | "enemy";

// Something bolts can hit: an upright cylinder at a ground position
export type Target = {
  team: Team;
  position: THREE.Vector3;
  radius: number;
  explodes: boolean;
  // Called with the bolt's level direction of travel
  hit: (direction: THREE.Vector3) => void;
};

type Bolt = {
  mesh: THREE.Mesh;
  light: THREE.PointLight;
  team: Team;
  origin: THREE.Vector3;
  direction: THREE.Vector3;
  // Distance of the front end from the origin; once burst, from the impact
  head: number;
  age: number;
  burst: number; // time since it burst, or -1 while flying
};



const yAxis = new THREE.Vector3(0, 1, 0);
const toTarget = new THREE.Vector2();
const impact = new THREE.Vector3();

export class BoltPool {
  readonly group = new THREE.Group();
  readonly targets = new Set<Target>();
  private readonly bolts: Bolt[] = [];
  // Enough rows along it for the splat to bend it smoothly
  private readonly geometry = new THREE.CapsuleGeometry(BOLT_RADIUS, BOLT_LENGTH, 4, 8, 40);
  private readonly material = new THREE.MeshBasicNodeMaterial();
  // Seen from inside too, once it splats
  private readonly boltMaterial = new THREE.MeshBasicNodeMaterial({ side: THREE.DoubleSide });
  private next = 0;
  private readonly time = uniform(0);
  private readonly sparkGeometry = new THREE.BoxGeometry(1, 1, 1);
  private readonly bloodGeometry = teardrop();
  private readonly bloodMaterial = new THREE.MeshBasicNodeMaterial({
    color: "#2a0306",
    transparent: true,
    opacity: 0.85,
  });
  private readonly mistGeometry = new THREE.IcosahedronGeometry(0.5, 2);
  // Soft edged: see-through towards its outline, so it reads as a cloud
  private readonly mistMaterial = new THREE.MeshBasicNodeMaterial({
    color: "#5a0710",
    transparent: true,
    depthWrite: false,
  });
  private readonly mist: Debris;
  private readonly sparks: Debris;
  private readonly blood: Debris;

  constructor(stains: Stains) {
    // Over-bright so it stays saturated deep red after tone mapping, and
    // flagged for the selective bloom pass by a zero alpha
    this.material.colorNode = vec3(4.5, 0.12, 0.075);
    this.material.mrtNode = mrt({ output: vec4(output.rgb, 0.0) });

    // The splat, in the bolt's own space (along y, front at +y): past each
    // bolt's cut, the impact, it folds back and spreads out, and is dropped
    // once it has flowed in SPLAT_REACH
    const cutAt = uniform(UNCUT).onObjectUpdate(({ object }) => object!.userData.cut);
    const flowed = max(positionLocal.y.sub(cutAt), 0.0);
    const across = length(positionLocal.xz);
    const spread = across.add(flowed.mul(SPLAT_SPREAD)).div(max(across, 1e-4));
    this.boltMaterial.positionNode = vec3(
      positionLocal.x.mul(spread),
      positionLocal.y.min(cutAt).sub(flowed.mul(SPLAT_BACK)),
      positionLocal.z.mul(spread)
    );
    const gone = varying(flowed).greaterThan(SPLAT_REACH);
    this.boltMaterial.colorNode = Fn(() => {
      If(gone, () => Discard());
      return this.material.colorNode!;
    })();
    this.boltMaterial.mrtNode = this.material.mrtNode;

    for (let i = 0; i < POOL_SIZE; i++) {
      const mesh = new THREE.Mesh(this.geometry, this.boltMaterial);
      mesh.visible = false;
      mesh.frustumCulled = false;
      mesh.userData.cut = UNCUT;
      const light = new THREE.PointLight(LIGHT_COLOR, 0, LIGHT_RANGE, 2);
      light.visible = false;

      this.group.add(mesh, light);
      this.bolts.push({
        mesh,
        light,
        team: "player",
        origin: new THREE.Vector3(),
        direction: new THREE.Vector3(),
        head: 0,
        age: BOLT_LIFE,
        burst: -1,
      });
    }

    // Gooey wobble: the surface bulges in and out with ripples that drift
    // through space, so blobs churn as they fly, on the pool's own clock,
    // which stops while the game is paused. The material sees each blob
    // already placed, so this works in world space, along its normals
    const time = this.time;
    const ripple = sin(positionLocal.x.mul(9.0).add(time.mul(14.0)))
      .mul(sin(positionLocal.y.mul(8.0).sub(time.mul(11.0))))
      .mul(sin(positionLocal.z.mul(7.0).add(time.mul(13.0))));
    this.bloodMaterial.positionNode = positionLocal.add(normalLocal.normalize().mul(ripple.mul(BLOOD_WOBBLE)));

    this.mistMaterial.opacityNode = pow(saturate(normalView.z), 2.0).mul(MIST_OPACITY);

    // Same glowing material, so sparks bloom too
    this.sparks = new Debris(SPARKS, this.sparkGeometry, this.material);
    // Blobs landing in the grass knock dents into it
    this.blood = new Debris(BLOOD, this.bloodGeometry, this.bloodMaterial, (position, width) =>
      stains.dent(position.x, position.z, width * DENT_SIZE)
    );
    this.mist = new Debris(MIST, this.mistGeometry, this.mistMaterial);
    this.group.add(this.sparks.mesh, this.blood.mesh, this.mist.mesh);
  }

  // Fire from a muzzle along a level direction, reusing the oldest bolt
  fire(team: Team, muzzle: THREE.Vector3, direction: THREE.Vector3) {
    const bolt = this.bolts[this.next];
    this.next = (this.next + 1) % this.bolts.length;
    bolt.team = team;
    bolt.age = 0;
    bolt.head = 0;
    bolt.burst = -1;
    bolt.light.distance = LIGHT_RANGE;
    bolt.light.visible = true;
    bolt.origin.copy(muzzle);
    bolt.direction.copy(direction).setY(0).normalize();
    bolt.mesh.quaternion.setFromUnitVectors(yAxis, bolt.direction);
    bolt.mesh.userData.cut = UNCUT;
    bolt.mesh.visible = true;
  }

  update(delta: number) {
    this.time.value += delta;
    for (const bolt of this.bolts) {
      if (bolt.burst >= 0) {
        this.updateBurst(bolt, delta);
        continue;
      }
      if (bolt.age >= BOLT_LIFE) continue;
      bolt.age += delta;
      if (bolt.age >= BOLT_LIFE) {
        this.end(bolt);
        continue;
      }
      const grown = Math.min(bolt.age / BOLT_GROW, 1);
      const ease = 1 - (1 - grown) * (1 - grown);
      const width = THREE.MathUtils.lerp(ORB_WIDTH, 1, ease);
      const length = THREE.MathUtils.lerp(ORB_LENGTH, 1, ease);
      bolt.mesh.scale.set(width, length, width);
      // Tail at the muzzle while growing, then the whole bolt flies on
      const travelled = Math.max(bolt.age - BOLT_GROW, 0) * BOLT_SPEED;
      const head = travelled + BOLT_LENGTH * length;

      // Hits anything of another team the front swept past this frame
      const target = this.sweep(bolt, bolt.head, head);
      if (target) {
        target.hit(bolt.direction);
        if (target.explodes) this.explode(bolt, target, head);
        else this.end(bolt);
        continue;
      }
      bolt.head = head;

      bolt.mesh.position
        .copy(bolt.origin)
        .addScaledVector(bolt.direction, travelled + (BOLT_LENGTH * length) / 2);
      bolt.light.position.copy(bolt.mesh.position);
      bolt.light.intensity = LIGHT_INTENSITY * (1 + MUZZLE_FLASH * (1 - ease));
    }
    this.sparks.update(delta);
    this.blood.update(delta);
    this.mist.update(delta);
  }

  // Turn the bolt into a burst where it met the target, its front head
  // along its way
  private explode(bolt: Bolt, target: Target, head: number) {
    const along =
      (target.position.x - bolt.origin.x) * bolt.direction.x +
      (target.position.z - bolt.origin.z) * bolt.direction.z;
    impact.copy(bolt.origin).addScaledVector(bolt.direction, along);
    bolt.burst = 0;
    bolt.origin.copy(impact);
    bolt.head = head - along;
    bolt.light.distance = BURST_LIGHT_RANGE;
    bolt.light.position.copy(impact);

    this.sparks.burst(impact, bolt.direction);
    this.blood.burst(impact, bolt.direction);
    this.mist.burst(impact, bolt.direction);
  }

  // The bolt flowing into the impact and splatting, and the light flashing
  // and dying
  private updateBurst(bolt: Bolt, delta: number) {
    bolt.burst += delta;
    if (bolt.burst >= BURST_TIME) {
      bolt.burst = -1;
      this.end(bolt);
      return;
    }
    bolt.light.intensity = BURST_LIGHT * (1 - bolt.burst / BURST_TIME) ** 2;

    // Its front this far past the impact, in metres and in the mesh's own
    // units along it (scaled by its length)
    const front = bolt.head + SPLAT_SPEED * bolt.burst;
    const length = bolt.mesh.scale.y;
    bolt.mesh.visible = front < BOLT_LENGTH * length + SPLAT_REACH;
    bolt.mesh.position.copy(bolt.origin).addScaledVector(bolt.direction, front - (BOLT_LENGTH * length) / 2);
    bolt.mesh.userData.cut = ((BOLT_LENGTH * length) / 2 - front) / length;
  }

  private sweep(bolt: Bolt, from: number, to: number) {
    for (const target of this.targets) {
      if (target.team === bolt.team) continue;
      toTarget.set(target.position.x - bolt.origin.x, target.position.z - bolt.origin.z);
      const along = toTarget.x * bolt.direction.x + toTarget.y * bolt.direction.z;
      if (along < from - target.radius || along > to + target.radius) continue;
      const across = toTarget.x * bolt.direction.z - toTarget.y * bolt.direction.x;
      if (Math.abs(across) > target.radius) continue;
      return target;
    }
    return null;
  }

  private end(bolt: Bolt) {
    bolt.age = BOLT_LIFE;
    bolt.mesh.visible = false;
    bolt.light.intensity = 0;
    bolt.light.visible = false;
  }

  dispose() {
    this.geometry.dispose();
    this.sparkGeometry.dispose();
    this.bloodGeometry.dispose();
    this.bloodMaterial.dispose();
    this.mistGeometry.dispose();
    this.mistMaterial.dispose();
    this.material.dispose();
    this.boltMaterial.dispose();
  }
}
