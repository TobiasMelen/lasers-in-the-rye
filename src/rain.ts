import * as THREE from "three/webgpu";
import { Fn, abs, float, floor, fract, length, smoothstep, step } from "three/tsl";
import { hash12, hash22, noise } from "./tslNoise";

// Raindrop rings on still water. Each cell of a world grid drops on its own
// random beat, landing at a new random spot every time and sending out one
// ring that widens and fades. The spot stays far enough from the cell edge
// for the ring to fit, so only the cell under the pixel is checked; more
// grids, offset and seeded differently, add drops across the first's edges

const CELL = 0.9; // metres
const MAX_RADIUS = 0.27; // metres, fits inside the cell from the spots below
const DROP_CHANCE = 0.9; // fraction of beats that actually drop

// Grid offsets (in cells) and hash seeds of the overlapping drop layers
const LAYERS: [number, number][] = [
  [0.0, 0.0],
  [0.5, 31.7],
  [0.25, -57.1],
  [0.75, 83.9],
];

const dropLayer = (
  xz: THREE.Node<"vec2">,
  time: THREE.Node<"float">,
  wobble: THREE.Node<"float">,
  offset: number,
  layerSeed: number
) => {
  const p = xz.div(CELL).add(offset);
  const cell = floor(p).add(layerSeed);
  const period = hash12(cell.add(19.7)).mul(0.4).add(0.5);
  const beat = time.div(period).add(hash12(cell.add(-7.3)));
  const age = fract(beat);
  // A new landing spot and chance of a drop on every beat
  const seed = cell.add(floor(beat).mul(0.37));
  const center = hash22(seed).mul(0.4).add(0.3);
  const falls = step(hash12(seed.add(5.1)), DROP_CHANCE);
  // Each drop gets its own size and strength
  const size = hash12(seed.add(-2.9)).mul(0.6).add(0.4);
  const strength = hash12(seed.add(8.3)).mul(0.7).add(0.3);
  // Break the ring's line up so it isn't a clean circle
  const distance = length(fract(p).sub(center)).mul(CELL).add(wobble);
  const ring = smoothstep(
    0.015,
    0.0,
    abs(distance.sub(age.mul(MAX_RADIUS).mul(size)))
  ).mul(smoothstep(0.15, 0.65, noise(xz.mul(40.0).add(seed))));
  const fade = age.oneMinus();
  return ring.mul(fade.mul(fade)).mul(falls).mul(strength);
};

// Ring brightness at a world xz position
export const raindrops = Fn(
  ([xz, time]: [THREE.Node<"vec2">, THREE.Node<"float">]) => {
    // Wobbles every ring's radius, the same in every layer
    const wobble = noise(xz.mul(18.0)).sub(0.5).mul(0.03).toVar();
    let rings: THREE.Node<"float"> = float(0.0);
    for (const [offset, layerSeed] of LAYERS) {
      rings = rings.add(dropLayer(xz, time, wobble, offset, layerSeed));
    }
    return rings;
  }
);
