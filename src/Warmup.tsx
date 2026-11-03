import { useThree } from "@react-three/fiber";
import { useEffect } from "react";
import * as THREE from "three/webgpu";

// Compiles every shader before play starts. Mounted once everything has
// loaded, it shows whatever is hidden (pooled enemies, idle bolts) for a few
// frames, so each material gets drawn, and its pipeline built, while the
// loading screen still covers the view. Then it hides them again and waits
// for the GPU to finish before reporting ready
const WARMUP_FRAMES = 3;

const nextFrame = () => new Promise((resolve) => requestAnimationFrame(resolve));

export default function Warmup({ onReady }: { onReady: () => void }) {
  const scene = useThree((state) => state.scene);
  const renderer = useThree((state) => state.gl) as unknown as THREE.WebGPURenderer;

  useEffect(() => {
    let cancelled = false;
    const revealed: THREE.Object3D[] = [];
    scene.traverse((object) => {
      if (!object.visible) {
        object.visible = true;
        revealed.push(object);
      }
    });
    const hide = () => {
      for (const object of revealed) object.visible = false;
      revealed.length = 0;
    };

    (async () => {
      for (let i = 0; i < WARMUP_FRAMES; i++) await nextFrame();
      const backend = renderer.backend as unknown as { device?: GPUDevice };
      await backend.device?.queue.onSubmittedWorkDone();
      if (cancelled) return;
      hide();
      onReady();
    })();

    return () => {
      cancelled = true;
      hide();
    };
  }, [scene, renderer, onReady]);

  return null;
}
