import { Suspense, useCallback, useRef, useState, useEffect, useLayoutEffect, useMemo } from "react";
import { Canvas, extend, useFrame, useLoader, useThree, type Catalogue } from "@react-three/fiber";
import * as THREE from "three/webgpu";
import {
  color,
  Fn,
  If,
  vec3,
  mix,
  normalMap,
  positionWorld,
  smoothstep,
  texture,
  uniform,
  max,
  vec2,
} from "three/tsl";
import { useControls } from "leva";
import WebGPUGrass, { type GrassLook } from "./WebGPUGrass";
import { ISO_OFFSET, ISO_UP, ISO_VIEW_HEIGHT } from "./iso";
import { noise } from "./tslNoise";
import { raindrops } from "./rain";
import Lasers, { SelectiveBloom, type AntiAliasing, type Aim } from "./Lasers";
import Character from "./Character";
import Enemies from "./Enemies";
import Warmup from "./Warmup";
import { BoltPool } from "./bolts";
import { Stains } from "./stains";
import { ClusteredLighting } from "./lighting/ClusteredLighting";
import { bareGround, pathInset } from "./worldMap";

// Make R3F intrinsic elements resolve to the WebGPU build of three
extend(THREE as unknown as Catalogue);

type ScenePreset = {
  background: string;
  fog?: { color: string; near: number; far: number };
  ambient: { color: string; intensity: number };
  light: {
    position: [number, number, number];
    color: string;
    intensity: number;
  };
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

function SceneLighting({ preset }: { preset: ScenePreset }) {
  return (
    <>
      <color attach="background" args={[preset.background]} />
      {preset.fog && (
        <fog
          attach="fog"
          args={[preset.fog.color, preset.fog.near, preset.fog.far]}
        />
      )}
      <ambientLight
        color={preset.ambient.color}
        intensity={preset.ambient.intensity}
      />
      <directionalLight
        position={preset.light.position}
        color={preset.light.color}
        intensity={preset.light.intensity}
      />
      {preset.moon && (
        <mesh position={preset.moon.position}>
          <sphereGeometry args={[preset.moon.radius, 32, 16]} />
          <meshBasicMaterial color={preset.moon.color} fog={false} />
        </mesh>
      )}
    </>
  );
}

// Brown Mud 03 by Rob Tuytel, Poly Haven (CC0). Scanned at 1.3 m, drawn
// larger so the cracks read at iso distance
const GRAVEL_TILE = 8;

const GRAVEL_URLS = [
  `${import.meta.env.BASE_URL}textures/brown_mud_03_diff_512.webp`,
  `${import.meta.env.BASE_URL}textures/brown_mud_03_nor_gl_512.webp`,
  // Packed ambient occlusion, roughness, metalness
  `${import.meta.env.BASE_URL}textures/brown_mud_03_arm_512.webp`,
];

function tiling(map: THREE.Texture, colorSpace: string) {
  map.wrapS = map.wrapT = THREE.RepeatWrapping;
  map.colorSpace = colorSpace;
  map.anisotropy = 8;
  map.needsUpdate = true;
  return map;
}

// Ground plane that travels with the ball, so the field never ends. Shows
// gravel where the world map has paths
function Ground({ groundRef, follow, lightColor }) {
  const light = useMemo(() => uniform(new THREE.Color()), []);
  const time = useMemo(() => uniform(0), []);
  useEffect(() => {
    light.value.set(lightColor);
  }, [light, lightColor]);

  const gravelMaps = useLoader(THREE.TextureLoader, GRAVEL_URLS);

  const material = useMemo(() => {
    const gravelColor = tiling(gravelMaps[0], THREE.SRGBColorSpace);
    const gravelNormal = tiling(gravelMaps[1], THREE.NoColorSpace);
    const gravelArm = tiling(gravelMaps[2], THREE.NoColorSpace);

    const mat = new THREE.MeshStandardNodeMaterial({ roughness: 1 });
    const xz = positionWorld.xz;
    // World xz mapped to follow the plane's own uv axes (v runs along -z),
    // so the normal map's tangent frame lines up
    const gravelUv = vec2(xz.x, xz.y.negate()).div(GRAVEL_TILE);
    const bare = bareGround(xz);

    const arm = texture(gravelArm, gravelUv);
    const occlusion = mix(1.0, arm.r, bare);

    // Random dark spots break up the flat dirt: damp blotches around half a
    // metre across, and small specks like pebble shadows
    const blotches = smoothstep(0.55, 0.75, noise(xz.mul(1.4).add(17.0)));
    const specks = smoothstep(0.7, 0.85, noise(xz.mul(9.0)));
    const spots = mix(1.0, 0.5, blotches).mul(mix(1.0, 0.6, specks));

    // Puddles in the middle of paths, ringed by darker wet dirt
    const puddleNoise = noise(xz.mul(0.45).add(vec2(-31.0, 8.0)));
    const deepEnough = smoothstep(0.35, 0.6, pathInset(xz));
    const puddle = smoothstep(0.65, 0.7, puddleNoise).mul(deepEnough);
    const wet = smoothstep(0.56, 0.68, puddleNoise).mul(deepEnough);

    // Broad light and dark patches hide the tiling
    const gravel = texture(gravelColor, gravelUv)
      .rgb.mul(noise(xz.mul(0.15)).mul(0.3).add(0.85))
      .mul(spots)
      .mul(mix(1.0, 0.55, wet));
    // Fake shadow from the grass: the path darkens towards its edges
    const grassShadow = mix(0.2, 1.0, pathInset(xz));
    // Occlusion also goes into the colour at half strength, since ambient
    // light alone (the only light aoNode affects) is too weak to make the
    // cracks read. Puddles are black smooth metal, so only sun glints and
    // the raindrop rings show on them
    const dirt = mix(color("#232712"), gravel.mul(grassShadow), bare).mul(
      mix(1.0, occlusion, 0.5)
    );
    mat.colorNode = mix(dirt, color("#000000"), puddle);
    mat.aoNode = occlusion;
    mat.metalnessNode = puddle;
    mat.roughnessNode = mix(mix(1.0, arm.g, bare), 0.05, max(puddle, wet.mul(0.5)));
    // Raindrop rings catch the moon/sunlight. Only worked out on puddles,
    // the rest of the ground skips them
    mat.emissiveNode = Fn(() => {
      const rings = vec3(0.0).toVar();
      If(puddle.greaterThan(0.0), () => {
        rings.assign(light.mul(raindrops(xz, time).mul(0.45)).mul(puddle));
      });
      return rings;
    })();
    // Doubled normal strength so the relief reads from the iso distance;
    // water lies flat
    mat.normalNode = normalMap(
      texture(gravelNormal, gravelUv),
      vec2(bare.mul(2.0).mul(puddle.oneMinus()))
    );
    return mat;
  }, [gravelMaps, light, time]);
  useEffect(() => () => material.dispose(), [material]);

  useFrame((state) => {
    time.value = state.clock.elapsedTime;
    groundRef.current?.position.set(follow.x, -0.01, follow.z);
  });

  return (
    <mesh ref={groundRef} rotation={[-Math.PI / 2, 0, 0]}>
      <planeGeometry args={[200, 200]} />
      <primitive object={material} attach="material" />
    </mesh>
  );
}

// Fixed isometric orthographic camera that glides after the character
const aimAhead = new THREE.Vector3();
const screenUp = new THREE.Vector3(ISO_UP[0], 0, ISO_UP[1]);
const view = new THREE.Vector3();

// Leads ahead of the character towards the mouse, so more of the view lies
// where it's looking. Eased slower than the follow, so it swings after a turn
const LOOK_AHEAD = 5; // metres
// Extra lead along the screen's vertical, which the view foreshortens
const LOOK_AHEAD_VERTICAL = 2.2;
const LOOK_AHEAD_RATE = 1.8; // per second
const FOLLOW_RATE = 4; // per second

function IsoCamera({ target, aim }: { target: THREE.Vector3; aim: Aim }) {
  const camera = useThree((state) => state.camera);
  const height = useThree((state) => state.size.height);
  const focus = useRef(target.clone());
  const lead = useRef(new THREE.Vector3());

  useLayoutEffect(() => {
    camera.zoom = height / ISO_VIEW_HEIGHT;
    camera.updateProjectionMatrix();
  }, [camera, height]);

  // Runs before the default frame callbacks, so the grass places its slot
  // grid for this frame's camera rather than last frame's
  useFrame((_, delta) => {
    focus.current.lerp(target, 1 - Math.exp(-delta * FOLLOW_RATE));
    lead.current.lerp(
      aimAhead
        .copy(aim.direction)
        .addScaledVector(
          screenUp,
          aim.direction.dot(screenUp) * (LOOK_AHEAD_VERTICAL - 1)
        )
        .multiplyScalar(LOOK_AHEAD),
      1 - Math.exp(-delta * LOOK_AHEAD_RATE)
    );
    view.copy(focus.current).add(lead.current);
    camera.position.set(...ISO_OFFSET).add(view);
    camera.lookAt(view);
  }, -1);

  return null;
}

// Covers the view while assets load and shaders compile, then fades out
function LoadingScreen({ visible }: { visible: boolean }) {
  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "#070b16",
        color: "#8fa0c8",
        font: "500 14px/1 ui-monospace, SFMono-Regular, Menlo, monospace",
        letterSpacing: "0.2em",
        textTransform: "uppercase",
        opacity: visible ? 1 : 0,
        pointerEvents: visible ? "auto" : "none",
        transition: "opacity 0.6s ease",
      }}
    >
      Loading…
    </div>
  );
}

// Drives the frames itself (the canvas runs frameloop "never"), skipping
// display refreshes to hold a frame rate cap: on 120 Hz displays, a 60 cap
// halves the GPU's work. 0 runs at the display's rate
function FrameLimiter({ fps }: { fps: number }) {
  const advance = useThree((state) => state.advance);
  const clock = useThree((state) => state.clock);
  useEffect(() => {
    // A little early is fine: refreshes never land exactly on the interval
    const interval = fps > 0 ? 1000 / fps - 2 : 0;
    let last = -Infinity;
    let id = requestAnimationFrame(function tick(now) {
      id = requestAnimationFrame(tick);
      if (burning || now - last < interval) return;
      // The clock runs on the timestamps given, in seconds; the first
      // frame starts it a frame back rather than from zero
      if (last === -Infinity && clock.elapsedTime === 0) clock.elapsedTime = now / 1000 - 1 / 60;
      last = now;
      advance(now / 1000);
    });
    return () => cancelAnimationFrame(id);
  }, [advance, clock, fps]);
  return null;
}

// BENCH: temporary. ?bench collects GPU/CPU frame times for window.__bench()
const BENCH = new URLSearchParams(location.search);
const BENCH_ON = BENCH.has("bench");
// Set while window.__burn drives the frames itself
let burning = false;

function Bench() {
  const gl = useThree((state) => state.gl) as unknown as THREE.WebGPURenderer;
  const advance = useThree((state) => state.advance);
  const clock = useThree((state) => state.clock);
  const scene = useThree((state) => state.scene);
  (window as unknown as { __scene: THREE.Scene }).__scene = scene;
  const samples = useRef({ render: [] as number[], compute: [] as number[], cpu: [] as number[] });
  const frameStart = useRef(0);
  const frame = useRef(0);
  const lastStart = useRef({ render: 0, compute: 0 });
  useFrame(() => {
    frameStart.current = performance.now();
  }, -2);
  useFrame(() => {
    const s = samples.current;
    s.cpu.push(performance.now() - frameStart.current);
    // A resolve covers every frame since the previous one started, all
    // tagged with the same frame id here, so average over those frames
    frame.current++;
    const pools = (gl.backend as unknown as {
      timestampQueryPool: Record<string, { timestamps: Map<string, number>; pendingResolve: unknown }>;
    }).timestampQueryPool;
    for (const type of ["render", "compute"] as const) {
      const pool = pools[type];
      if (!pool || pool.pendingResolve) continue;
      const frames = frame.current - lastStart.current[type];
      lastStart.current[type] = frame.current;
      gl.resolveTimestampsAsync(type).then(() => {
        let total = 0;
        for (const ms of pool.timestamps.values()) total += ms;
        if (frames > 0 && total > 0) s[type].push(total / frames);
      });
    }
  }, 2);
  useEffect(() => {
    const median = (a: number[]) => (a.length ? [...a].sort((x, y) => x - y)[a.length >> 1] : 0);
    (window as unknown as { __bench: () => object }).__bench = () => {
      const s = samples.current;
      const result = {
        frames: s.cpu.length,
        gpuRender: +median(s.render).toFixed(2),
        gpuCompute: +median(s.compute).toFixed(2),
        cpu: +median(s.cpu).toFixed(2),
        dpr: gl.getPixelRatio(),
        size: [gl.domElement.width, gl.domElement.height],
        samples: gl.samples,
      };
      s.render.length = s.compute.length = s.cpu.length = 0;
      return result;
    };
    // Renders frames back to back, with no wait for the display, at a
    // fixed 60 fps game step, keeping two frames in flight on the GPU
    (window as unknown as { __burn: (frames?: number) => Promise<object> }).__burn = async (frames = 600) => {
      const device = (gl.backend as unknown as { device: GPUDevice }).device;
      burning = true;
      await device.queue.onSubmittedWorkDone();
      let time = clock.elapsedTime;
      let previous: Promise<undefined> = Promise.resolve(undefined);
      const start = performance.now();
      for (let i = 0; i < frames; i++) {
        time += 1 / 60;
        advance(time);
        const done = device.queue.onSubmittedWorkDone();
        await previous;
        previous = done;
      }
      await previous;
      const ms = performance.now() - start;
      clock.elapsedTime = performance.now() / 1000 - 1 / 60;
      burning = false;
      return { frames, ms: +(ms / frames).toFixed(2), fps: Math.round((frames * 1000) / ms) };
    };
  }, [gl, advance, clock]);
  return null;
}

export default function App() {
  // Mutated every frame by the character; everything else reads it
  const [playerPosition] = useState(() => new THREE.Vector3());
  const [aim] = useState<Aim>(() => ({
    active: false,
    direction: new THREE.Vector3(0, 0, 1),
    shots: 0,
    muzzle: new THREE.Vector3(),
  }));
  // Everything that flattens the grass: the player, and enemies while around
  const [grassPushers] = useState(() => new Set([playerPosition]));
  const [stains] = useState(() => new Stains());
  const [bolts] = useState(() => new BoltPool(stains));
  const groundRef = useRef();
  // Until every asset has loaded and every shader compiled, a loading
  // screen covers the view and enemies hold off
  const [ready, setReady] = useState(false);
  const onReady = useCallback(() => setReady(true), []);

  // Project description in Leva
  const {
    Scene: sceneName,
    Resolution: resolution,
    "Anti-aliasing": antiAliasing,
    "Frame rate cap": frameRateCap,
  } = useControls("Controls", {
    "Controls": {
      value: "WASD: Move, Mouse: Shoot",
      editable: false
    },
    Scene: { value: "Moon", options: Object.keys(SCENES) },
    // Dev settings. On a 2x display, 1.5x resolution and no multisampling
    // each take a large share of the GPU work
    Resolution: { value: Number(BENCH.get("res") ?? 2), options: { "2x": 2, "1.5x": 1.5, "1x": 1 } },
    "Anti-aliasing": {
      value: (BENCH.get("aa") ?? "msaa") as AntiAliasing,
      options: {
        "MSAA 4x": "msaa",
        SMAA: "smaa",
        TRAA: "traa",
        "TAAU (render 1/2)": "taau50",
        "TAAU (render 3/4)": "taau75",
        Off: "off",
      } as Record<string, AntiAliasing>,
    },
    "Frame rate cap": { value: Number(BENCH.get("fps") ?? 60), options: { "60": 60, Display: 0 } },
  });
  const scene = SCENES[sceneName];

  return (
    <div style={{ width: "100vw", height: "100vh" }}>
      {/* flat: no tone mapping, so values above 1 reach HDR displays */}
      <Canvas
        flat
        orthographic
        frameloop="never"
        dpr={Math.min(resolution, window.devicePixelRatio)}
        camera={{ position: ISO_OFFSET, near: 1, far: 200 }}
        style={{
          userSelect: "none",
          WebkitUserSelect: "none",
          touchAction: "none"
        }}
        gl={async ({ canvas }) => {
          const renderer = new THREE.WebGPURenderer({
            canvas: canvas as HTMLCanvasElement,
            // Multisampling is the scene pass's (see SelectiveBloom); the
            // canvas only gets the finished image
            antialias: false,
            trackTimestamp: BENCH_ON,
            alpha: false,
            powerPreference: "high-performance",
            // Half-float canvas with extended tone mapping: HDR output on
            // displays and browsers that support it, plain SDR elsewhere
            outputType: THREE.HalfFloatType,
          });
          // Point lights (the laser bolts) only shade the pixels they reach
          renderer.lighting = new ClusteredLighting();
          await renderer.init();
          return renderer;
        }}
      >
        <SceneLighting key={sceneName} preset={scene} />

        <WebGPUGrass spherePosition={playerPosition} pushers={grassPushers} stains={stains} look={scene.grass} sunPosition={scene.light.position} />
        <Lasers origin={playerPosition} aim={aim} bolts={bolts} />
        <Suspense fallback={null}>
          <Ground
            groundRef={groundRef}
            follow={playerPosition}
            lightColor={scene.light.color}
          />
          <Character position={playerPosition} aim={aim} bolts={bolts} />
          <Enemies
            player={playerPosition}
            bolts={bolts}
            grassPushers={grassPushers}
            stains={stains}
            active={ready}
          />
          {/* Last, so it mounts once everything above has loaded */}
          <Warmup onReady={onReady} />
        </Suspense>
        <SelectiveBloom antiAliasing={antiAliasing} />

        <IsoCamera target={playerPosition} aim={aim} />
        <FrameLimiter fps={frameRateCap} />
        {BENCH_ON && <Bench />}
      </Canvas>
      <LoadingScreen visible={!ready} />
    </div>
  );
}
