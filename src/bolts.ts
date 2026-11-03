import * as THREE from "three/webgpu";
import {
  float,
  instanceIndex,
  mix,
  mrt,
  normalLocal,
  normalView,
  positionLocal,
  pow,
  saturate,
  sin,
  smoothstep,
  time,
  uv,
  vec3,
} from "three/tsl";
import { Debris, type DebrisStyle } from "./debris";
import type { Stains } from "./stains";

// Red laser bolts, shared by everyone who shoots. Each flies level from
// where it was fired and ends on the first target of another team it
// passes through, or after BOLT_LIFE. A hit on something that explodes
// slams the bolt flat into a flying disc and flashes its light, with a
// cone of fire, sparks and blood blown on the way the bolt was going

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
// BURST_TIME. The bolt itself slams flat like it hit a wall, squashing into
// a wide thin disc across its path, a nail's head, which keeps flying on at
// the bolt's speed, spreading wider as it thins away to nothing
const BURST_TIME = 0.25; // seconds
const BURST_LIGHT = 240; // light intensity at the flash
const BURST_LIGHT_RANGE = 16;
const HEAD_SLAM = 0.03; // seconds to squash flat
const HEAD_FADE = 0.16; // seconds until it's gone
const HEAD_RADIUS = 1.1; // metres across once flat
const HEAD_SPREAD = 1.8; // times wider by the time it's gone
const HEAD_THICKNESS = 0.22; // metres along its path once flat

// Cone of fire opening from the impact along the bolt's way, shooting out
// to full length fast and fading from its tip back
// Each burst's cone is a random size up to these, and tipped a little off
// the bolt's line
const CONE_LENGTH = 4; // metres
const CONE_RADIUS = 1.5; // metres at its open end
const CONE_MIN_SIZE = 0.5; // smallest, as a fraction of the full size
const CONE_JITTER = 0.18; // radians, either way
const CONE_TIME = 0.12; // seconds

// Sparks: hot streaks blown on in a cone along the bolt's way, slowing and
// falling
const SPARKS: DebrisStyle = {
  pool: 320,
  perBurst: 64,
  life: [0.3, 0.65],
  speed: [10, 24],
  carry: 2.6,
  fan: 0.45,
  lift: 0.1,
  gravity: 22,
  drag: 1.2,
  width: 0.08,
  stretch: 0.065,
  shrink: 1,
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
const MIST_OPACITY = 0.19;

// How far blood blobs' surfaces bulge in and out as they fly, in metres
const BLOOD_WOBBLE = 0.07;

// A blob's dent in the grass, as a multiple of its width
const DENT_SIZE = 2.2;

// Blood mist: a faint red cloud puffed out of the body, swelling as it
// rolls out the bolt's way over the splatter as that spreads, slowing as
// it goes, and thins out
const MIST: DebrisStyle = {
  pool: 48,
  perBurst: 8,
  life: [0.9, 1.4],
  speed: [8, 16],
  carry: 3,
  fan: 0.7,
  lift: 0.15,
  gravity: 1,
  drag: 2,
  width: 1.3,
  variety: 0.4,
  stretch: 0,
  shrink: 0.6,
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
  head: number; // distance of the front end from the origin
  age: number;
  burst: number; // time since it burst, or -1 while flying
  index: number; // its instance in the cone mesh
  coneRotation: THREE.Quaternion;
  coneLength: number;
  coneRadius: number;
};



const yAxis = new THREE.Vector3(0, 1, 0);
const toTarget = new THREE.Vector2();
const impact = new THREE.Vector3();
const coneAxis = new THREE.Vector3();
const coneMatrix = new THREE.Matrix4();
const coneScale = new THREE.Vector3();
const coneGlow = new THREE.Color();
const hidden = new THREE.Matrix4().makeScale(0, 0, 0);

export class BoltPool {
  readonly group = new THREE.Group();
  readonly targets = new Set<Target>();
  private readonly bolts: Bolt[] = [];
  private readonly geometry = new THREE.CapsuleGeometry(BOLT_RADIUS, BOLT_LENGTH, 4, 8);
  private readonly material = new THREE.MeshBasicNodeMaterial();
  private next = 0;
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
  private readonly coneGeometry = new THREE.ConeGeometry(1, 1, 24, 1, true);
  // Every cone in one instanced draw: glowing, added on top, brightest at
  // the apex and fading to nothing at the open end. Each fades on its own
  // through its instance colour
  private readonly coneMaterial = new THREE.MeshBasicNodeMaterial({
    transparent: true,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  private readonly cones: THREE.InstancedMesh;
  private readonly sparks: Debris;
  private readonly blood: Debris;

  constructor(stains: Stains) {
    // Over-bright so it stays saturated deep red after tone mapping, and
    // flagged for the selective bloom pass
    this.material.colorNode = vec3(3.0, 0.08, 0.05);
    this.material.mrtNode = mrt({ bloomIntensity: float(1.0) });

    // Apex at the origin, opening along +y, so it scales to length and width
    this.coneGeometry.rotateX(Math.PI);
    this.coneGeometry.translate(0, 0.5, 0);
    // Broken into ragged jets of fire rather than an even sheet: brightness
    // and reach vary around it, in a pattern that churns over time and
    // differs per cone. Whole waves around, so there's no seam
    const around = uv().x.mul(Math.PI * 2);
    const churn = time.mul(9.0).add(float(instanceIndex).mul(2.7));
    const jets = saturate(
      sin(around.mul(5.0).add(churn))
        .mul(sin(around.mul(8.0).sub(churn.mul(1.3))))
        .add(sin(around.mul(3.0).add(churn.mul(0.7))).mul(0.5))
        .mul(0.6)
        .add(0.5)
    );
    const jetReach = sin(around.mul(7.0).sub(churn.mul(0.8)))
      .mul(sin(around.mul(4.0).add(churn.mul(1.6))))
      .mul(0.3)
      .add(0.7);
    const reached = smoothstep(jetReach, jetReach.sub(0.2), uv().y.oneMinus());
    this.coneMaterial.colorNode = vec3(3.0, 0.09, 0.05)
      .mul(pow(uv().y, 2.2))
      .mul(mix(0.2, 1.0, jets))
      .mul(reached);
    this.coneMaterial.mrtNode = mrt({ bloomIntensity: float(1.0) });
    this.cones = new THREE.InstancedMesh(this.coneGeometry, this.coneMaterial, POOL_SIZE);
    this.cones.frustumCulled = false;

    for (let i = 0; i < POOL_SIZE; i++) {
      const mesh = new THREE.Mesh(this.geometry, this.material);
      mesh.visible = false;
      mesh.frustumCulled = false;
      const light = new THREE.PointLight(LIGHT_COLOR, 0, LIGHT_RANGE, 2);
      light.visible = false;

      this.group.add(mesh, light);
      this.cones.setMatrixAt(i, hidden);
      this.cones.setColorAt(i, coneGlow.setScalar(0));
      this.bolts.push({
        mesh,
        light,
        team: "player",
        origin: new THREE.Vector3(),
        direction: new THREE.Vector3(),
        head: 0,
        age: BOLT_LIFE,
        burst: -1,
        index: i,
        coneRotation: new THREE.Quaternion(),
        coneLength: CONE_LENGTH,
        coneRadius: CONE_RADIUS,
      });
    }

    // Gooey wobble: the surface bulges in and out with ripples that drift
    // through space, so blobs churn as they fly. The material sees each blob
    // already placed, so this works in world space, along its normals
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
    this.group.add(this.cones, this.sparks.mesh, this.blood.mesh, this.mist.mesh);
  }

  // Fire from a muzzle along a level direction, reusing the oldest bolt
  fire(team: Team, muzzle: THREE.Vector3, direction: THREE.Vector3) {
    const bolt = this.bolts[this.next];
    this.next = (this.next + 1) % this.bolts.length;
    bolt.team = team;
    if (bolt.burst >= 0) this.hideCone(bolt);
    bolt.age = 0;
    bolt.head = 0;
    bolt.burst = -1;
    bolt.light.distance = LIGHT_RANGE;
    bolt.light.visible = true;
    bolt.origin.copy(muzzle);
    bolt.direction.copy(direction).setY(0).normalize();
    bolt.mesh.quaternion.setFromUnitVectors(yAxis, bolt.direction);
    bolt.mesh.visible = true;
  }

  update(delta: number) {
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
        if (target.explodes) this.explode(bolt, target);
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

  // Turn the bolt into a burst where it met the target
  private explode(bolt: Bolt, target: Target) {
    const along =
      (target.position.x - bolt.origin.x) * bolt.direction.x +
      (target.position.z - bolt.origin.z) * bolt.direction.z;
    impact.copy(bolt.origin).addScaledVector(bolt.direction, along);
    bolt.burst = 0;
    bolt.origin.copy(impact);
    bolt.light.distance = BURST_LIGHT_RANGE;
    coneAxis
      .copy(bolt.direction)
      .applyAxisAngle(yAxis, THREE.MathUtils.randFloatSpread(CONE_JITTER * 2));
    coneAxis.y = THREE.MathUtils.randFloatSpread(CONE_JITTER);
    bolt.coneRotation.setFromUnitVectors(yAxis, coneAxis.normalize());
    // Random size, and a little random in its proportions
    const size = THREE.MathUtils.randFloat(CONE_MIN_SIZE, 1);
    bolt.coneLength = CONE_LENGTH * size;
    bolt.coneRadius = CONE_RADIUS * size * THREE.MathUtils.randFloat(0.75, 1);

    this.sparks.burst(impact, bolt.direction);
    this.blood.burst(impact, bolt.direction);
    this.mist.burst(impact, bolt.direction);
  }

  // The cone of fire, the bolt slammed into a flying head, and the light
  // flashing and dying
  private updateBurst(bolt: Bolt, delta: number) {
    bolt.burst += delta;
    if (bolt.burst >= BURST_TIME) {
      bolt.burst = -1;
      this.hideCone(bolt);
      this.end(bolt);
      return;
    }

    // The cone shoots out to full length, widening, as it fades
    const shot = Math.min(bolt.burst / CONE_TIME, 1);
    const reach = 1 - (1 - shot) ** 3;
    if (shot < 1) {
      coneScale.set(bolt.coneRadius * reach, bolt.coneLength * reach, bolt.coneRadius * reach);
      this.cones.setMatrixAt(bolt.index, coneMatrix.compose(bolt.origin, bolt.coneRotation, coneScale));
      this.cones.setColorAt(bolt.index, coneGlow.setScalar((1 - shot) ** 3));
      this.cones.instanceMatrix.needsUpdate = true;
      this.cones.instanceColor!.needsUpdate = true;
    } else {
      this.hideCone(bolt);
    }
    // Slam: bolt length and thickness to the flat head's, then thin away
    // while spreading. Sizes along the path include the capsule's caps
    const slam = 1 - (1 - Math.min(bolt.burst / HEAD_SLAM, 1)) ** 2;
    const fade = Math.min(Math.max((bolt.burst - HEAD_SLAM) / (HEAD_FADE - HEAD_SLAM), 0), 1);
    const fullLength = BOLT_LENGTH + BOLT_RADIUS * 2;
    const length = THREE.MathUtils.lerp(fullLength, HEAD_THICKNESS, slam) * (1 - fade);
    const radius =
      THREE.MathUtils.lerp(BOLT_RADIUS, HEAD_RADIUS, slam) * THREE.MathUtils.lerp(1, HEAD_SPREAD, fade);
    bolt.mesh.visible = length > 0;
    bolt.mesh.scale.set(radius / BOLT_RADIUS, length / fullLength, radius / BOLT_RADIUS);
    // Its front flies on from the impact (kept in origin) at the bolt's speed
    bolt.mesh.position
      .copy(bolt.origin)
      .addScaledVector(bolt.direction, BOLT_SPEED * bolt.burst - length / 2);
    // The flash sits a little way into the cone
    bolt.light.intensity = BURST_LIGHT * (1 - bolt.burst / BURST_TIME) ** 2;
    bolt.light.position.copy(bolt.origin).addScaledVector(bolt.direction, bolt.coneLength * 0.25);
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

  private hideCone(bolt: Bolt) {
    this.cones.setMatrixAt(bolt.index, hidden);
    this.cones.instanceMatrix.needsUpdate = true;
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
    this.coneGeometry.dispose();
    this.material.dispose();
    this.coneMaterial.dispose();
  }
}
