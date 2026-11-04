import { useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three/webgpu";
import {
  acesFilmicToneMapping,
  float,
  luminance,
  max,
  mix,
  mrt,
  output,
  pass,
  pow,
  renderOutput,
  saturate,
  sRGBTransferOETF,
  vec3,
  vec4,
  velocity,
} from "three/tsl";
import { bloom } from "three/examples/jsm/tsl/display/BloomNode.js";
import { smaa } from "three/examples/jsm/tsl/display/SMAANode.js";
import { traa } from "three/examples/jsm/tsl/display/TRAANode.js";
import { taau } from "three/examples/jsm/tsl/display/TAAUNode.js";
import type { BoltPool } from "./bolts";

// Until the character reports its rifle's muzzle, bolts aim on this plane
const AIM_HEIGHT = 1.2;
const FIRE_INTERVAL = 0.12; // seconds between shots while the button is held

// Where the mouse aims, shared with the character. Lasers writes the
// direction from the character (for it to face), active while the button is
// held and shots counts bolts fired (for its recoil); the character writes
// its rifle's muzzle, where bolts leave from
export type Aim = {
  active: boolean;
  direction: THREE.Vector3;
  shots: number;
  muzzle: THREE.Vector3;
};

const aimPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -AIM_HEIGHT);
const raycaster = new THREE.Raycaster();
const aimPoint = new THREE.Vector3();
const boltDirection = new THREE.Vector3();

// Fires the player's bolts from the character towards the mouse while the
// mouse button is held, and flies everyone's bolts
export default function Lasers({
  origin,
  aim,
  bolts,
}: {
  origin: THREE.Vector3;
  aim: Aim;
  bolts: BoltPool;
}) {
  const canvas = useThree((state) => state.gl.domElement);
  const firing = useRef(false);
  // Mouse position in normalised device coordinates, tracked here rather
  // than through R3F's pointer so presses aim where they happen
  const pointer = useRef(new THREE.Vector2());
  const cooldown = useRef(0);

  useEffect(() => () => bolts.dispose(), [bolts]);

  useEffect(() => {
    const aim = (event: PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
      pointer.current.set(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1
      );
    };
    const start = (event: PointerEvent) => {
      aim(event);
      if (event.button === 0) firing.current = true;
    };
    const stop = () => {
      firing.current = false;
    };
    canvas.addEventListener("pointerdown", start);
    window.addEventListener("pointermove", aim);
    window.addEventListener("pointerup", stop);
    return () => {
      canvas.removeEventListener("pointerdown", start);
      window.removeEventListener("pointermove", aim);
      window.removeEventListener("pointerup", stop);
    };
  }, [canvas]);

  useFrame((state, delta) => {
    cooldown.current = Math.max(cooldown.current - delta, 0);

    // Aim on the plane at muzzle height, so bolts fly level through the
    // point under the mouse
    const muzzle = aim.muzzle;
    aimPlane.constant = -(muzzle.y > 0 ? muzzle.y : AIM_HEIGHT);
    raycaster.setFromCamera(pointer.current, state.camera);
    const hit = raycaster.ray.intersectPlane(aimPlane, aimPoint);
    if (hit) {
      const direction = boltDirection.subVectors(aimPoint, origin).setY(0);
      if (direction.lengthSq() > 1e-4) aim.direction.copy(direction.normalize());
    }
    aim.active = firing.current;

    if (firing.current && cooldown.current === 0) {
      // From the muzzle to the mouse, unless the mouse is too close to it
      const direction = boltDirection.copy(aim.direction);
      if (hit) {
        const toMouse = aimPoint.clone().sub(muzzle).setY(0);
        if (toMouse.lengthSq() > 1) direction.copy(toMouse.normalize());
      }
      bolts.fire("player", muzzle, direction);
      cooldown.current = FIRE_INTERVAL;
      aim.shots++;
    }

    bolts.update(delta);
  });

  return <primitive object={bolts.group} />;
}

// Takes over rendering: only what the bolts' materials flag blooms. The
// flag rides in the scene colour's alpha, unused otherwise, rather than a
// target of its own: opaque surfaces and the background write 1, the bolts
// 0, so the mask is its inverse. Tone mapping is done here:
// ACES Filmic for the normal range, with what lies above 1 added back on top
// so bright highlights reach HDR displays
const HIGHLIGHT_KEEP = 0.6;
// Final night grade, before tone mapping. Most colour drains to a blue
// monochrome (SATURATION 1 keeps the colours as rendered, 0 is all tint),
// then contrast steepens around a mid grey and exposure darkens it
const SATURATION = 0.66;
const TINT = [0.83, 0.93, 1.29] as const;
const CONTRAST = 1.09;
const CONTRAST_PIVOT = 0.18;
const EXPOSURE = 0.97;

// TAAU renders the scene at a fraction of the output resolution (half or
// three quarters across) and builds the full one up over frames
export type AntiAliasing = "msaa" | "smaa" | "traa" | "taau50" | "taau75" | "off";

const TAAU_SCALE: Partial<Record<AntiAliasing, number>> = { taau50: 0.5, taau75: 0.75 };

export function SelectiveBloom({ antiAliasing }: { antiAliasing: AntiAliasing }) {
  const renderer = useThree((state) => state.gl) as unknown as THREE.WebGPURenderer;
  const scene = useThree((state) => state.scene);
  const camera = useThree((state) => state.camera);

  const pipeline = useMemo(() => {
    // Multisampled here if at all: the scene is the only pass with edges
    const samples = antiAliasing === "msaa" ? 4 : 0;
    const scenePass = pass(scene, camera, { samples });
    // The temporal ones also need each pixel's motion since last frame
    const upscale = TAAU_SCALE[antiAliasing];
    const temporal = antiAliasing === "traa" || upscale !== undefined;
    scenePass.setMRT(temporal ? mrt({ output, velocity }) : mrt({ output }));
    if (upscale !== undefined) scenePass.setResolutionScale(upscale);
    const resolve = upscale !== undefined ? taau : traa;
    const color = temporal
      ? resolve(
          scenePass.getTextureNode(),
          scenePass.getTextureNode("depth"),
          scenePass.getTextureNode("velocity"),
          camera
        )
      : scenePass.getTextureNode();
    const mask = saturate(scenePass.getTextureNode().a.oneMinus());
    const glow = bloom(color.mul(mask), 2.0, 0.6, 0.0);
    // Quarter resolution (default half): the glow is soft anyway, and this
    // makes every blur step about 4x cheaper
    glow.setResolutionScale(0.25);
    // Grade everything but the bolts, so they stay pure red
    const base = color.rgb;
    const blue = vec3(luminance(base)).mul(vec3(...TINT));
    const toned = mix(blue, base, SATURATION);
    const contrasted = pow(max(toned, 0.0).div(CONTRAST_PIVOT), vec3(CONTRAST))
      .mul(CONTRAST_PIVOT * EXPOSURE);
    const graded = mix(contrasted, base, mask);
    const hdr = graded.add(glow.rgb);
    const filmic = acesFilmicToneMapping(hdr, float(1.0));
    const highlights = max(hdr.sub(1.0), 0.0).mul(HIGHLIGHT_KEEP);

    // To sRGB here rather than in renderOutput, so SMAA finds edges in
    // perceptual values: in linear ones the dark night scene has almost none
    const display = vec4(sRGBTransferOETF(filmic.add(highlights)), 1.0);
    const pipeline = new THREE.RenderPipeline(renderer);
    pipeline.outputColorTransform = false;
    pipeline.outputNode = renderOutput(
      antiAliasing === "smaa" ? smaa(display) : display,
      THREE.NoToneMapping,
      THREE.LinearSRGBColorSpace
    );
    return pipeline;
  }, [renderer, scene, camera, antiAliasing]);
  useEffect(() => () => pipeline.dispose(), [pipeline]);

  useFrame(() => pipeline.render(), 1);

  return null;
}
