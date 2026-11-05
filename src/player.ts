import * as THREE from "three/webgpu";
import { ISO_RIGHT, ISO_UP } from "./iso";
import { Bot, type BotAssets } from "./bot";
import type { BoltPool } from "./bolts";

// The player's bot: WASD moves it along the screen's axes, it always aims
// at the mouse, and fires bolts towards it while the button is held

const MAX_SPEED = 4.5; // metres per second
const ACCELERATION = 8; // how quickly velocity reaches the input, per second
const FIRE_INTERVAL = 0.12; // seconds between shots while the button is held
// Until the rifle reports its muzzle, bolts aim on this plane
const AIM_HEIGHT = 1.2;

// How close an enemy bolt has to pass to hit
const HIT_RADIUS = 0.55;

// Dark Souls style hero light: a soft warm glow given off by the character,
// from its chest, lighting the ground around it a little. Decay below the
// physical 2, so it fades out more gently and covers more ground
const HERO_LIGHT_OFFSET = new THREE.Vector3(0, 1.4, 0);

const aimPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -AIM_HEIGHT);
const raycaster = new THREE.Raycaster();
const aimPoint = new THREE.Vector3();
const boltDirection = new THREE.Vector3();
const move = new THREE.Vector3();

export function createPlayer(assets: BotAssets, bolts: BoltPool, canvas: HTMLCanvasElement) {
  const bot = new Bot(assets, "silver");
  // Where it looks, shared with the camera, which leads that way
  const aim = new THREE.Vector3(0, 0, 1);
  const heroLight = new THREE.PointLight("#ffe6c7", 2.4, 12, 1.5);
  const object = new THREE.Group().add(bot.object, heroLight);
  const keys = new Set<string>();
  // Mouse position in normalised device coordinates
  const pointer = new THREE.Vector2();
  let firing = false;
  let cooldown = 0;

  // Enemy bolts stop on the player; there's no damage yet
  bolts.targets.add({ team: "player", position: bot.position, radius: HIT_RADIUS, explodes: false, hit: () => {} });

  window.addEventListener("keydown", (event) => keys.add(event.code));
  window.addEventListener("keyup", (event) => keys.delete(event.code));
  const point = (event: PointerEvent) => {
    const rect = canvas.getBoundingClientRect();
    pointer.set(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1
    );
  };
  window.addEventListener("pointermove", point);
  canvas.addEventListener("pointerdown", (event) => {
    point(event);
    if (event.button === 0) firing = true;
  });
  window.addEventListener("pointerup", () => (firing = false));

  function update(delta: number, camera: THREE.Camera) {
    // Aim on the plane at muzzle height, so bolts fly level through the
    // point under the mouse
    const muzzle = bot.muzzle;
    aimPlane.constant = -(muzzle.y > 0 ? muzzle.y : AIM_HEIGHT);
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.ray.intersectPlane(aimPlane, aimPoint);
    if (hit) {
      const direction = boltDirection.subVectors(aimPoint, bot.position).setY(0);
      if (direction.lengthSq() > 1e-4) aim.copy(direction.normalize());
    }

    cooldown = Math.max(cooldown - delta, 0);
    if (firing && cooldown === 0) {
      // From the muzzle to the mouse, unless the mouse is too close to it
      const direction = boltDirection.copy(aim);
      if (hit) {
        const toMouse = aimPoint.sub(muzzle).setY(0);
        if (toMouse.lengthSq() > 1) direction.copy(toMouse.normalize());
      }
      bolts.fire("player", muzzle, direction);
      cooldown = FIRE_INTERVAL;
      bot.recoil();
    }

    // Input along the screen's ground axes
    const inputUp = +keys.has("KeyW") - +keys.has("KeyS");
    const inputRight = +keys.has("KeyD") - +keys.has("KeyA");
    move.set(
      ISO_UP[0] * inputUp + ISO_RIGHT[0] * inputRight,
      0,
      ISO_UP[1] * inputUp + ISO_RIGHT[1] * inputRight
    );
    if (move.lengthSq() > 0) move.normalize().multiplyScalar(MAX_SPEED);
    bot.velocity.lerp(move, 1 - Math.exp(-delta * ACCELERATION));

    bot.update(delta, aim);
    heroLight.position.copy(bot.position).add(HERO_LIGHT_OFFSET);
  }

  return { object, position: bot.position, aim, update };
}
