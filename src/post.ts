import * as THREE from "three/webgpu";
import {
  acesFilmicToneMapping,
  agxToneMapping,
  cineonToneMapping,
  float,
  linearToneMapping,
  luminance,
  max,
  mix,
  mrt,
  neutralToneMapping,
  output,
  pass,
  pow,
  reinhardToneMapping,
  renderOutput,
  saturate,
  sRGBTransferOETF,
  vec3,
  uniform,
  vec4,
} from "three/tsl";
import { bloom } from "three/examples/jsm/tsl/display/BloomNode.js";

// Renders the scene: only what the bolts' materials flag blooms. The flag
// rides in the scene colour's alpha, unused otherwise, rather than a target
// of its own: opaque surfaces and the background write 1, the bolts 0, so
// the mask is its inverse. Tone mapping is done here: one of TONE_MAPPINGS
// for the normal range, with what lies above 1 added back on top so bright
// highlights reach HDR displays. On an HDR canvas all of it is kept, and
// the bolts and sparks (whatever blooms) get hdrBoost times their brightest
// channel on top again, tinted HDR_CORE: an orange, white-hot core inside
// their red glow. hdrBoost is a setting, changed live
export const TONE_MAPPINGS = {
  "ACES Filmic": acesFilmicToneMapping,
  AgX: agxToneMapping,
  Neutral: neutralToneMapping,
  Reinhard: reinhardToneMapping,
  Cineon: cineonToneMapping,
  // None: clips at white
  Linear: linearToneMapping,
};
const HIGHLIGHT_KEEP = 0.6;
const HDR_HIGHLIGHT_KEEP = 1.0;
export const hdrBoost = uniform(8.0);
const HDR_CORE = [1.0, 0.4, 0.12] as const;
// Glow strength and spread (0 to 1), more of both on an HDR canvas, which
// has room above white
const BLOOM = 2.0;
const HDR_BLOOM = 3.0;
const BLOOM_RADIUS = 0.6;
const HDR_BLOOM_RADIUS = 0.8;
// Final night grade, before tone mapping. Most colour drains to a blue
// monochrome (SATURATION 1 keeps the colours as rendered, 0 is all tint),
// then contrast steepens around a mid grey and exposure darkens it
const SATURATION = 0.66;
const TINT = [0.83, 0.93, 1.29] as const;
const CONTRAST = 1.09;
const CONTRAST_PIVOT = 0.18;
const EXPOSURE = 0.97;

export function createPost(
  renderer: THREE.WebGPURenderer,
  scene: THREE.Scene,
  camera: THREE.Camera,
  // Multisampled here if at all: the scene is the only pass with edges
  msaa: boolean,
  // The canvas takes values past white (see main)
  hdrOutput: boolean,
  // A TONE_MAPPINGS name, ACES Filmic if unknown
  toneMapping: string
) {
  const scenePass = pass(scene, camera, { samples: msaa ? 4 : 0 });
  // Materials (the bolts') merge their own outputs into this
  scenePass.setMRT(mrt({ output }));
  const color = scenePass.getTextureNode();
  const mask = saturate(color.a.oneMinus());
  const glow = bloom(
    color.mul(mask),
    hdrOutput ? HDR_BLOOM : BLOOM,
    hdrOutput ? HDR_BLOOM_RADIUS : BLOOM_RADIUS,
    0.0
  );
  // Quarter resolution (default half): the glow is soft anyway, and this
  // makes every blur step about 4x cheaper
  glow.setResolutionScale(0.25);
  // Grade everything but the bolts, so they stay pure red
  const base = color.rgb;
  const blue = vec3(luminance(base)).mul(vec3(...TINT));
  const toned = mix(blue, base, SATURATION);
  const contrasted = pow(max(toned, 0.0).div(CONTRAST_PIVOT), vec3(CONTRAST))
    .mul(CONTRAST_PIVOT * EXPOSURE);
  const hdr = mix(contrasted, base, mask).add(glow.rgb);
  const curve = TONE_MAPPINGS[toneMapping as keyof typeof TONE_MAPPINGS] ?? acesFilmicToneMapping;
  const filmic = curve(hdr, float(1.0));
  const over = max(hdr.sub(1.0), 0.0);
  const whiteHot = vec3(max(base.r, max(base.g, base.b))).mul(mask).mul(vec3(...HDR_CORE));
  const highlights = vec3(
    hdrOutput ? over.mul(HDR_HIGHLIGHT_KEEP).add(whiteHot.mul(hdrBoost)) : over.mul(HIGHLIGHT_KEEP)
  );

  const pipeline = new THREE.RenderPipeline(renderer);
  pipeline.outputColorTransform = false;
  pipeline.outputNode = renderOutput(
    vec4(sRGBTransferOETF(filmic.add(highlights)), 1.0),
    THREE.NoToneMapping,
    THREE.LinearSRGBColorSpace
  );
  return pipeline;
}
