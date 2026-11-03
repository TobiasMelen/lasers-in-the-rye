import { useLoader } from "@react-three/fiber";
import { useMemo } from "react";
import * as THREE from "three/webgpu";
import { mix, output, positionWorld, smoothstep, vec4 } from "three/tsl";
import { GLTFLoader, type GLTF } from "three/examples/jsm/loaders/GLTFLoader.js";
import { MeshoptDecoder } from "three/examples/jsm/libs/meshopt_decoder.module.js";
import { clone as cloneSkinned } from "three/examples/jsm/utils/SkeletonUtils.js";

// Mixamo's Y Bot in polished metal (silver or gold) with a placeholder
// rifle, driven by Mixamo rifle clips. Both are baked from the Mixamo FBX
// downloads, outside the app, into one simplified, meshopt compressed GLB.
// Always aims along a given direction; the legs blend forward, back and
// strafe clips by how it moves relative to that. Shared by the player and
// the enemies. Scaled up so it reads above the grass
const MODEL_URL = `${import.meta.env.BASE_URL}models/bot.glb`;
const MODEL_SCALE = 1.4;
const SOURCE_UNITS = 0.01; // the Mixamo source is in centimetres

const TURN_RATE = 14; // how quickly the body turns to face, per second

// Placeholder rifle: a level block from behind the right hand out past the
// left, which the clips hold it with. Its far end is where bolts leave from
const RIFLE_LENGTH = 1.5;
const RIFLE_STOCK = 0.4; // how far it reaches back behind the right hand
const RIFLE_RAISE = 0.04; // centre above the grips, so it rests in the fists
const RIFLE_WIDTH = 0.08;
const RIFLE_HEIGHT = 0.13;

// The clips hold the rifle turned off the body's forward, so the body turns
// by however much that is to put the rifle on the aim. Eased, so the pose's
// sway doesn't jitter it, and capped in case the hands swap over
const AIM_CORRECTION_RATE = 10; // per second
const AIM_CORRECTION_MAX = 0.8; // radians

// Bones the firing recoil leaves alone, so it only moves the upper body
const LOWER_BODY = /Hips|Leg|Foot|Toe/;

const rightHand = new THREE.Vector3();
const leftHand = new THREE.Vector3();
const rifleAxis = new THREE.Vector3();
const bonePosition = new THREE.Vector3();

// World position of a grip: the midpoint of its bones, raised to the
// rifle's centre
function gripPosition(bones: THREE.Object3D[], target: THREE.Vector3) {
  target.set(0, 0, 0);
  for (const bone of bones) target.add(bone.getWorldPosition(bonePosition));
  return target.divideScalar(bones.length).setY(target.y + RIFLE_RAISE);
}

// Shortest signed angle from a to b
export const angleTo = (a: number, b: number) =>
  THREE.MathUtils.euclideanModulo(b - a + Math.PI, Math.PI * 2) - Math.PI;

// What the silver reflects: a small equirect night sky, darker blue at the
// top than at the horizon, a near black ground and the moon, in the same
// direction as the Moon scene's light
const MOON_DIRECTION = new THREE.Vector3(-49, 50, 7).normalize();

function nightReflections() {
  const width = 128;
  const height = 64;
  const data = new Float32Array(width * height * 4);
  const direction = new THREE.Vector3();
  for (let y = 0; y < height; y++) {
    // Rows run from straight down (v = 0) to straight up
    const elevation = ((y + 0.5) / height - 0.5) * Math.PI;
    for (let x = 0; x < width; x++) {
      // Same mapping as three's equirectUV
      const azimuth = ((x + 0.5) / width - 0.5) * Math.PI * 2;
      direction.set(
        Math.cos(elevation) * Math.cos(azimuth),
        Math.sin(elevation),
        Math.cos(elevation) * Math.sin(azimuth)
      );
      const up = Math.max(direction.y, 0);
      const sky = Math.pow(1 - up, 3);
      let r = 0.012 + 0.05 * sky;
      let g = 0.018 + 0.07 * sky;
      let b = 0.045 + 0.12 * sky;
      if (direction.y < 0) {
        // Ground: dark, a hint of grass
        const fade = Math.min(-direction.y * 6, 1);
        r = THREE.MathUtils.lerp(r, 0.006, fade);
        g = THREE.MathUtils.lerp(g, 0.012, fade);
        b = THREE.MathUtils.lerp(b, 0.012, fade);
      }
      const moon = direction.angleTo(MOON_DIRECTION);
      const glow = 8 * Math.exp(-(moon * moon) / 0.004) + 0.25 * Math.exp(-(moon * moon) / 0.08);
      const i = (y * width + x) * 4;
      data[i] = r + glow * 0.9;
      data[i + 1] = g + glow * 0.95;
      data[i + 2] = b + glow;
      data[i + 3] = 1;
    }
  }
  const texture = new THREE.DataTexture(data, width, height, THREE.RGBAFormat, THREE.FloatType);
  texture.mapping = THREE.EquirectangularReflectionMapping;
  texture.magFilter = THREE.LinearFilter;
  texture.needsUpdate = true;
  return texture;
}

// Darkening of the robot towards its feet: GROUND_SHADE at the ground up to
// none by GROUND_FADE_HEIGHT (metres; bots always stand at y = 0)
const GROUND_SHADE = 0.15;
const GROUND_FADE_HEIGHT = 0.9;
const groundFade = mix(
  GROUND_SHADE,
  1.0,
  smoothstep(0.0, GROUND_FADE_HEIGHT, positionWorld.y)
);

// Polished shell over darker joints. Metal shows only what it reflects, so
// both get a night sky to reflect
const FINISHES = {
  silver: { shell: "#d9dde3", joints: "#2b2e34" },
  gold: { shell: "#e8b04a", joints: "#3a2c18" },
};
export type Finish = keyof typeof FINISHES;

type Materials = { shell: THREE.Material; joints: THREE.Material };

function finishMaterials(finish: Finish, reflections: THREE.Texture): Materials {
  const make = (color: string, joints: boolean) => {
    const material = new THREE.MeshStandardNodeMaterial({
      color,
      metalness: joints ? 0.9 : 1.0,
      roughness: joints ? 0.35 : 0.18,
      envMap: reflections,
      // Brighter than the real sky so it still reads as metal at night
      envMapIntensity: 2.5,
    });
    // Fade the feet and lower legs into the dark grass
    material.outputNode = vec4(output.rgb.mul(groundFade), output.a);
    return material;
  };
  const colors = FINISHES[finish];
  return { shell: make(colors.shell, false), joints: make(colors.joints, true) };
}

export type BotAssets = {
  model: THREE.Group;
  materials: Record<Finish, Materials>;
  clips: THREE.AnimationClip[];
  speeds: Record<string, number>;
  recoil: THREE.AnimationClip;
  rifleGeometry: THREE.BufferGeometry;
  rifleMaterial: THREE.Material;
};

// Built once per loaded file, since every bot shares its materials
const assetCache = new WeakMap<GLTF, BotAssets>();

function buildAssets(gltf: GLTF): BotAssets {
  // The baked file's root: the model, with the clips' authored ground
  // speeds (before MODEL_SCALE) in its extras
  const model = gltf.scene.children[0] as THREE.Group;
  const speeds = model.userData.speeds as Record<string, number>;
  const reflections = nightReflections();
  const materials = {
    silver: finishMaterials("silver", reflections),
    gold: finishMaterials("gold", reflections),
  };
  model.traverse((object) => {
    if (!(object instanceof THREE.Mesh)) return;
    // Skinned bounds don't follow the animation, so don't cull on them
    object.frustumCulled = false;
    for (const old of [object.material].flat()) old.dispose();
  });

  const clips = gltf.animations;
  // Recoil as an offset from the clip's first frame, layered on top
  const recoil = THREE.AnimationClip.findByName(clips, "fire").clone();
  recoil.tracks = recoil.tracks.filter((track) => !LOWER_BODY.test(track.name));
  THREE.AnimationUtils.makeClipAdditive(recoil);

  const rifleGeometry = new THREE.BoxGeometry(RIFLE_WIDTH, RIFLE_HEIGHT, RIFLE_LENGTH);
  // Along +z from its origin at the stock, so lookAt points it
  rifleGeometry.translate(0, 0, RIFLE_LENGTH / 2);
  const rifleMaterial = new THREE.MeshStandardNodeMaterial({
    color: "#2a2d33",
    metalness: 0.5,
    roughness: 0.4,
  });

  return { model, materials, clips, speeds, recoil, rifleGeometry, rifleMaterial };
}

export function useBotAssets(): BotAssets {
  const gltf = useLoader(GLTFLoader, MODEL_URL, (loader) =>
    loader.setMeshoptDecoder(MeshoptDecoder)
  );
  return useMemo(() => {
    let assets = assetCache.get(gltf);
    if (!assets) {
      assets = buildAssets(gltf);
      assetCache.set(gltf, assets);
    }
    return assets;
  }, [gltf]);
}

type Gait = { action: THREE.AnimationAction; stride: number; speed: number };

export class Bot {
  // Add this to the scene: it holds the posed model and the rifle, which is
  // placed in world space
  readonly object = new THREE.Group();
  // Feet on the ground; set velocity, and update moves it
  readonly position: THREE.Vector3;
  readonly velocity = new THREE.Vector3();
  // Where bolts leave the rifle, after each update
  readonly muzzle = new THREE.Vector3();

  private readonly body = new THREE.Group();
  private readonly rifle: THREE.Mesh;
  private readonly mixer: THREE.AnimationMixer;
  private readonly grips: THREE.Object3D[][];
  private readonly idle: THREE.AnimationAction;
  private readonly walk: Gait;
  private readonly run: Gait;
  private readonly left: Gait;
  private readonly right: Gait;
  private readonly fire: THREE.AnimationAction;
  private yaw = 0;
  private aimCorrection = 0;
  // Shared cycle position of the locomotion clips, so feet stay in step
  // while they blend
  private phase = 0;

  constructor(assets: BotAssets, finish: Finish, position = new THREE.Vector3()) {
    this.position = position;
    const model = cloneSkinned(assets.model);
    const materials = assets.materials[finish];
    model.traverse((object) => {
      if (object instanceof THREE.Mesh) {
        object.material = object.name.endsWith("_Joints") ? materials.joints : materials.shell;
      }
    });
    model.scale.setScalar(MODEL_SCALE * SOURCE_UNITS);
    this.body.add(model);
    this.rifle = new THREE.Mesh(assets.rifleGeometry, assets.rifleMaterial);
    this.object.add(this.body, this.rifle);

    this.mixer = new THREE.AnimationMixer(model);
    const clip = (name: string) => THREE.AnimationClip.findByName(assets.clips, name);
    // Locomotion clips are posed by setting their time directly from phase
    const gait = (name: string): Gait => {
      const action = this.mixer.clipAction(clip(name));
      action.timeScale = 0;
      const speed = assets.speeds[name] * MODEL_SCALE;
      return { action, stride: speed * clip(name).duration, speed };
    };
    this.idle = this.mixer.clipAction(clip("idle"));
    this.walk = gait("walk");
    this.run = gait("run");
    this.left = gait("strafeLeft");
    this.right = gait("strafeRight");
    this.fire = this.mixer.clipAction(assets.recoil);
    this.fire.setLoop(THREE.LoopOnce, 1);
    for (const action of [this.idle, this.walk.action, this.run.action, this.left.action, this.right.action]) {
      action.play();
    }

    // Each grip sits between the knuckles and the thumb, inside the fist
    this.grips = (["Right", "Left"] as const).map((side) =>
      ["Middle1", "Thumb2"].map(
        (bone) => model.getObjectByName(`mixamorig${side}Hand${bone}`)!
      )
    );
  }

  // Snap to face a direction, for a freshly placed bot
  face(direction: THREE.Vector3) {
    this.yaw = Math.atan2(direction.x, direction.z) + this.aimCorrection;
  }

  // Layer one firing recoil over the current pose
  recoil() {
    this.fire.reset().play();
  }

  // Moves by velocity, turns to aim along a direction, poses and places the
  // rifle
  update(delta: number, aim: THREE.Vector3) {
    this.position.addScaledVector(this.velocity, delta).setY(0);
    this.body.position.copy(this.position);

    const facing = Math.atan2(aim.x, aim.z) + this.aimCorrection;
    this.yaw += angleTo(this.yaw, facing) * (1 - Math.exp(-delta * TURN_RATE));
    this.body.rotation.y = this.yaw;

    // Movement in the body's frame: forward along the aim, left across it
    const velocity = this.velocity;
    const speed = velocity.length();
    const sin = Math.sin(this.yaw);
    const cos = Math.cos(this.yaw);
    const forward = sin * velocity.x + cos * velocity.z;
    const side = cos * velocity.x - sin * velocity.z;

    // Direction weights summing to 1 (squared cosines), forward and back
    // sharing the walk/run clips played in reverse for back
    const { walk, run, left, right } = this;
    const along = speed > 1e-3 ? (forward * forward) / (speed * speed) : 1;
    const strafe = 1 - along;
    const moving = THREE.MathUtils.clamp(speed / walk.speed, 0, 1);
    const running = THREE.MathUtils.clamp((speed - walk.speed) / (run.speed - walk.speed), 0, 1);
    const weights: [Gait, number][] = [
      [walk, along * (1 - running)],
      [run, along * running],
      [side > 0 ? left : right, strafe],
      [side > 0 ? right : left, 0],
    ];

    // Advance the shared phase at the blended clips' cadence for this speed
    let rate = 0;
    for (const [gait, weight] of weights) rate += (weight * speed) / gait.stride;
    this.phase = (this.phase + rate * delta) % 1;
    const backwards = forward < 0;

    this.idle.setEffectiveWeight(1 - moving);
    for (const [gait, weight] of weights) {
      gait.action.setEffectiveWeight(weight * moving);
      const reversed = backwards && (gait === walk || gait === run);
      gait.action.time = (reversed ? 1 - this.phase : this.phase) * gait.action.getClip().duration;
    }

    this.mixer.update(delta);

    // Lay the rifle along the hands and report its muzzle
    this.body.updateMatrixWorld(true);
    gripPosition(this.grips[0], rightHand);
    gripPosition(this.grips[1], leftHand);
    rifleAxis.subVectors(leftHand, rightHand).setY(0).normalize();
    this.rifle.position.copy(rightHand).addScaledVector(rifleAxis, -RIFLE_STOCK);
    this.rifle.lookAt(leftHand.copy(this.rifle.position).add(rifleAxis));

    const miss = angleTo(Math.atan2(rifleAxis.x, rifleAxis.z), Math.atan2(aim.x, aim.z));
    this.aimCorrection = THREE.MathUtils.clamp(
      this.aimCorrection + miss * (1 - Math.exp(-delta * AIM_CORRECTION_RATE)),
      -AIM_CORRECTION_MAX,
      AIM_CORRECTION_MAX
    );
    this.muzzle.copy(this.rifle.position).addScaledVector(rifleAxis, RIFLE_LENGTH);
  }
}
