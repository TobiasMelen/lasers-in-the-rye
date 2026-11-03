import * as THREE from "three/webgpu";

// Blood splats left on the grass where enemies die, which the grass paints
// dark red, thrown out the way the killing bolt was going. Each spreads out
// from the impact over a moment, as the blood lands, and is only painted
// while it spreads, so a ring of the most recent ones is enough. Also the
// dents blood blobs knock into the grass where they land, kept until the
// grass has stamped them

export const MAX_STAINS = 4;
export const MAX_DENTS = 8;
// Where unused stains sit, far from any grass
const NOWHERE = 1e6;

// Scale of the spray: its cone reaches about three of these
const RADIUS: [number, number] = [1.3, 2.0]; // metres, random between

// How far a stain has spread, in its radii: from just behind the impact to
// past its furthest droplets, easing out over this many seconds
const SPREAD_FROM = -0.5;
const SPREAD_TO = 7.5;
const SPREAD_TIME = 1.1;

export class Stains {
  // Each spot: world x, z, radius and the angle it was thrown towards
  // (atan2 of the direction's z and x), which also seeds its shape
  readonly spots = Array.from(
    { length: MAX_STAINS },
    () => new THREE.Vector4(NOWHERE, NOWHERE, 0, 0)
  );
  // How far each spot has spread, in its radii along its throw
  readonly reach = new Array<number>(MAX_STAINS).fill(SPREAD_TO);
  // Whether each spot spread this frame, and so needs painting
  readonly spreading = new Array<boolean>(MAX_STAINS).fill(false);
  private readonly ages = new Array<number>(MAX_STAINS).fill(SPREAD_TIME);
  private next = 0;

  // Dents waiting for the grass: x, z and radius
  readonly dents = Array.from(
    { length: MAX_DENTS },
    () => new THREE.Vector3(NOWHERE, NOWHERE, 0)
  );
  private nextDent = 0;

  add(x: number, z: number, direction: THREE.Vector3) {
    const radius = THREE.MathUtils.randFloat(...RADIUS);
    this.spots[this.next].set(x, z, radius, Math.atan2(direction.z, direction.x));
    this.ages[this.next] = 0;
    this.reach[this.next] = SPREAD_FROM;
    this.next = (this.next + 1) % MAX_STAINS;
  }

  dent(x: number, z: number, radius: number) {
    this.dents[this.nextDent].set(x, z, radius);
    this.nextDent = (this.nextDent + 1) % MAX_DENTS;
  }

  update(delta: number) {
    for (let i = 0; i < MAX_STAINS; i++) {
      this.spreading[i] = this.ages[i] < SPREAD_TIME;
      if (!this.spreading[i]) continue;
      this.ages[i] = Math.min(this.ages[i] + delta, SPREAD_TIME);
      const t = 1 - (1 - this.ages[i] / SPREAD_TIME) ** 2;
      this.reach[i] = THREE.MathUtils.lerp(SPREAD_FROM, SPREAD_TO, t);
    }
  }

  // Once the grass has its dents, they're done
  clearDents() {
    for (const dent of this.dents) dent.set(NOWHERE, NOWHERE, 0);
  }
}
