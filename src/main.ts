import * as THREE from "three/webgpu";
import GUI from "three/examples/jsm/libs/lil-gui.module.min.js";
import Renderer from "three/src/renderers/common/Renderer.js";
import WebGPUBackend from "three/src/renderers/webgpu/WebGPUBackend.js";
import StandardNodeLibrary from "three/src/renderers/webgpu/nodes/StandardNodeLibrary.js";
import { grassField, grassSettings, type GrassLook } from "./grass";
import { createGround } from "./ground";
import { createPlayer } from "./player";
import { createEnemies } from "./enemies";
import { createPost, hdrBoost, TONE_MAPPINGS } from "./post";
import { loadBotAssets } from "./bot";
import { BoltPool } from "./bolts";
import { Stains } from "./stains";
import { ClusteredLighting } from "./lighting/ClusteredLighting";
import { ISO_OFFSET, ISO_UP, ISO_VIEW_HEIGHT } from "./iso";

type ScenePreset = {
  background: string;
  fog?: { color: string; near: number; far: number };
  ambient: { color: string; intensity: number };
  light: { position: [number, number, number]; color: string; intensity: number };
  moon?: { position: [number, number, number]; radius: number; color: string };
  grass: GrassLook;
};

const SCENES: Record<string, ScenePreset> = {
  Day: {
    background: "#c2e2ff",
    ambient: { color: "#ffffff", intensity: 0.4 },
    // From the upper left of the iso view, low enough to rake across the
    // path relief and partly behind the grass for the backlit glow
    light: { position: [-10, 7, 1.4], color: "#FFFFCC", intensity: 1 },
    grass: { specular: "#55554a", shininess: 30, tipTint: [1.0, 0.9, 0.6], translucency: 0.35, wetness: 0 },
  },
  Moon: {
    background: "#070b16",
    fog: { color: "#070b16", near: 60, far: 150 },
    ambient: { color: "#6e80d8", intensity: 1.25 },
    // Same general direction as the Day sun
    light: { position: [-49, 50, 7], color: "#b4d6c4", intensity: 1.45 },
    moon: { position: [-245, 250, 35], radius: 9, color: "#eef2ff" },
    grass: { specular: "#b8ccc0", shininess: 20, tipTint: [0.75, 0.85, 1.0], translucency: 0, wetness: 1 },
  },
};

// The camera leads ahead of the character towards the mouse, so more of
// the view lies where it's looking. Eased slower than the follow, so it
// swings after a turn
const LOOK_AHEAD = 5; // metres
// Extra lead along the screen's vertical, which the view foreshortens
const LOOK_AHEAD_VERTICAL = 2.2;
const LOOK_AHEAD_RATE = 1.8; // per second
const FOLLOW_RATE = 4; // per second

// Dev settings. On a 2x display, 1.5x resolution and no multisampling each
// take a large share of the GPU work. A 60 fps cap on a 120 Hz display
// halves the GPU's work; 0 runs at the display's rate. HDR output costs
// about a third of the frame, and is off for good on screens that can't
// show it. Kept in localStorage, with the grass's
const hdrScreen = matchMedia("(dynamic-range: high)").matches;
const settings = {
  scene: "Moon",
  resolution: 2,
  msaa: true,
  fps: 60,
  hdr: hdrScreen,
  hdrBoost: hdrBoost.value,
  toneMapping: "ACES Filmic",
};
const SAVED = "settings";
try {
  const saved = JSON.parse(localStorage.getItem(SAVED) ?? "{}");
  Object.assign(settings, saved.settings);
  Object.assign(grassSettings, saved.grass);
} catch {
  // Nothing saved, or storage blocked: defaults
}
settings.hdr &&= hdrScreen;
hdrBoost.value = settings.hdrBoost;
const save = () => {
  try {
    localStorage.setItem(SAVED, JSON.stringify({ settings, grass: grassSettings }));
  } catch {
    // Storage blocked: settings last until reload
  }
};

function outputType() {
  return settings.hdr ? THREE.HalfFloatType : THREE.UnsignedByteType;
}

// three's WebGPURenderer, minus its WebGL fallback: the grass needs
// compute shaders, which WebGL lacks
const rendererOptions = {
  // Multisampling is the scene pass's (see post); the canvas only gets the
  // finished image
  antialias: false,
  alpha: false,
  powerPreference: "high-performance",
  // Half-float canvas with extended tone mapping: HDR output on displays
  // and browsers that support it, plain SDR elsewhere
  outputType: outputType(),
} as const;
const renderer = new Renderer(new WebGPUBackend(rendererOptions), rendererOptions) as THREE.WebGPURenderer;
renderer.library = new StandardNodeLibrary();
// Point lights (the laser bolts) only shade the pixels they reach
renderer.lighting = new ClusteredLighting();
await renderer.init();
document.body.prepend(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.OrthographicCamera();
camera.near = 1;
camera.far = 200;

const ambient = new THREE.AmbientLight();
const sun = new THREE.DirectionalLight();
const moon = new THREE.Mesh(
  new THREE.SphereGeometry(1, 32, 16),
  new THREE.MeshBasicNodeMaterial({ fog: false })
);
scene.add(ambient, sun, moon);

const stains = new Stains();
const bolts = new BoltPool(stains);
const [ground, botAssets] = await Promise.all([createGround(), loadBotAssets()]);
const player = createPlayer(botAssets, bolts, renderer.domElement);
// Everything that flattens the grass: the player, and enemies while around
const grassPushers = new Set([player.position]);
const enemies = createEnemies(botAssets, bolts, grassPushers, stains, player.position);
const grass = grassField(renderer, stains, grassPushers, player.position);
scene.add(ground.mesh, grass.mesh, player.object, enemies.object, bolts.group);
const makePost = () => createPost(renderer, scene, camera, settings.msaa, settings.hdr, settings.toneMapping);
let post = makePost();
function rebuildPost() {
  post.dispose();
  post = makePost();
}

// Switches between HDR and plain output. three sets up a canvas once per
// canvas target, reading the backend's output type, so this changes that
// and hands the same canvas over in a new target. Only the last pass draws
// to the canvas, so it's the only one rebuilt
function setOutput() {
  (renderer.backend as unknown as { parameters: { outputType: number } }).parameters.outputType = outputType();
  const old = renderer.getCanvasTarget();
  renderer.setCanvasTarget(new THREE.CanvasTarget(old.domElement));
  old.dispose();
  sizeCanvas();
  rebuildPost();
  boostControl.disable(!settings.hdr);
}

function applyScene() {
  const preset = SCENES[settings.scene];
  scene.background = new THREE.Color(preset.background);
  scene.fog = preset.fog ? new THREE.Fog(preset.fog.color, preset.fog.near, preset.fog.far) : null;
  ambient.color.set(preset.ambient.color);
  ambient.intensity = preset.ambient.intensity;
  sun.position.fromArray(preset.light.position);
  sun.color.set(preset.light.color);
  sun.intensity = preset.light.intensity;
  ground.light.value.set(preset.light.color);
  moon.visible = !!preset.moon;
  if (preset.moon) {
    moon.position.fromArray(preset.moon.position);
    moon.scale.setScalar(preset.moon.radius);
    (moon.material as THREE.MeshBasicNodeMaterial).color.set(preset.moon.color);
  }
}

function sizeCanvas() {
  renderer.setPixelRatio(Math.min(settings.resolution, window.devicePixelRatio));
  renderer.setSize(innerWidth, innerHeight);
}

function resize() {
  sizeCanvas();
  const aspect = innerWidth / innerHeight;
  camera.top = ISO_VIEW_HEIGHT / 2;
  camera.bottom = -camera.top;
  camera.right = camera.top * aspect;
  camera.left = -camera.right;
  camera.updateProjectionMatrix();
  grass.rebuild(aspect);
}

applyScene();
resize();
window.addEventListener("resize", resize);

// Bottom right, closed until opened. The game pauses while it's open
const gui = new GUI({ title: "Settings" }).onChange(save).close();
let paused = false;
gui.$title.addEventListener("click", () => (paused = !gui._closed));
Object.assign(gui.domElement.style, { top: "auto", bottom: "0" });
gui.add(settings, "scene", Object.keys(SCENES)).name("Scene").onChange(applyScene);
const graphicsGui = gui.addFolder("Graphics");
graphicsGui.add(settings, "resolution", { "2x": 2, "1.5x": 1.5, "1x": 1 }).name("Resolution").onChange(resize);
graphicsGui.add(settings, "msaa").name("MSAA 4x").onChange(rebuildPost);
graphicsGui.add(settings, "fps", { "60": 60, Display: 0 }).name("Frame rate cap");
graphicsGui.add(settings, "toneMapping", Object.keys(TONE_MAPPINGS)).name("Tone Mapping").onChange(rebuildPost);
graphicsGui.add(settings, "hdr").name("HDR").disable(!hdrScreen).onChange(setOutput);
// How bright the bolts' cores, and their glow, burn past white, on HDR only
const boostControl = graphicsGui
  .add(settings, "hdrBoost", 0, 16, 0.5)
  .name("HDR Boost")
  .disable(!settings.hdr)
  .onChange((value: number) => (hdrBoost.value = value));
const grassGui = gui.addFolder("Grass Field");
grassGui.add(grassSettings, "density", 10, 200, 5).name("Density").onFinishChange(resize);
grassGui.add(grassSettings, "bladeHeight", 0.7, 4, 0.1).name("Blade Height").onFinishChange(resize);
grassGui.add(grassSettings, "windSpeed", 0, 5, 0.1).name("Wind Speed");
grassGui.add(grassSettings, "trailRecovery", 0.2, 20, 0.1).name("Trail Recovery");
grassGui.add(grassSettings, "ridge", 0, 1, 0.05).name("Blade Ridge");
grassGui.add(grassSettings, "bladeLit").name("Blade Lighting").onChange(resize);

// Until every shader has compiled, the loading screen covers the view and
// enemies hold off
let ready = false;
let elapsed = 0;
const focus = new THREE.Vector3();
const lead = new THREE.Vector3();
const screenUp = new THREE.Vector3(ISO_UP[0], 0, ISO_UP[1]);
const view = new THREE.Vector3();

// Paused, everything holds still but still gets drawn, so settings changes
// show
function frame(delta: number) {
  if (paused) delta = 0;
  else {
    elapsed += delta;
    player.update(delta, camera);
    if (ready) enemies.update(delta);
    bolts.update(delta);
  }

  // Glide after the character, leading towards the mouse
  focus.lerp(player.position, 1 - Math.exp(-delta * FOLLOW_RATE));
  view
    .copy(player.aim)
    .addScaledVector(screenUp, player.aim.dot(screenUp) * (LOOK_AHEAD_VERTICAL - 1))
    .multiplyScalar(LOOK_AHEAD);
  lead.lerp(view, 1 - Math.exp(-delta * LOOK_AHEAD_RATE));
  view.copy(focus).add(lead);
  camera.position.set(...ISO_OFFSET).add(view);
  camera.lookAt(view);
  camera.updateMatrixWorld();

  ground.mesh.position.set(player.position.x, -0.01, player.position.z);
  ground.time.value = elapsed;
  const preset = SCENES[settings.scene];
  grass.update(delta, elapsed, scene, camera, preset.grass, preset.light.position);
  post.render();
}

// Driven by display refreshes, skipping some to hold the frame rate cap
let last = performance.now();
let burning = false;
requestAnimationFrame(function tick(now) {
  requestAnimationFrame(tick);
  // A little early is fine: refreshes never land exactly on the interval
  if (burning || now - last < (settings.fps ? 1000 / settings.fps - 2 : 0)) return;
  frame(Math.min((now - last) / 1000, 0.1));
  last = now;
});

// Compiles every shader before play starts: shows whatever is hidden
// (pooled enemies, idle bolts) for a few frames, so each material gets
// drawn while the loading screen still covers the view
const device = (renderer.backend as unknown as { device: GPUDevice }).device;
const hidden: THREE.Object3D[] = [];
scene.traverse((object) => {
  if (!object.visible) hidden.push(object);
});
for (const object of hidden) object.visible = true;
for (let i = 0; i < 3; i++) await new Promise(requestAnimationFrame);
await device.queue.onSubmittedWorkDone();
for (const object of hidden) object.visible = false;
ready = true;
document.getElementById("loading")!.style.opacity = "0";

// ?bench: window.__burn(frames) renders frames back to back, with no wait
// for the display, at a fixed 60 fps step, keeping two in flight on the GPU
if (new URLSearchParams(location.search).has("bench")) {
  Object.assign(window, {
    __scene: scene,
    __burn: async (frames = 600) => {
      burning = true;
      await device.queue.onSubmittedWorkDone();
      let previous = Promise.resolve();
      const start = performance.now();
      for (let i = 0; i < frames; i++) {
        frame(1 / 60);
        const done = device.queue.onSubmittedWorkDone();
        await previous;
        previous = done;
      }
      await previous;
      const ms = (performance.now() - start) / frames;
      last = performance.now();
      burning = false;
      return { ms: +ms.toFixed(2), fps: Math.round(1000 / ms) };
    },
  });
}
