import { defineConfig } from 'vite'
import { fileURLToPath } from 'node:url'

// three's own source modules rather than its prebuilt bundles, so the
// bundler drops what's unused (the WebGL fallback among it)
const three = (path: string) => fileURLToPath(new URL(`./node_modules/three/src/${path}`, import.meta.url))

export default defineConfig({
  // Relative, so the build works from any path (GitHub Pages serves it
  // under /<repo>/)
  base: './',
  resolve: {
    alias: [
      { find: /^three\/src\/(.*)$/, replacement: three('$1') },
      { find: /^three\/tsl$/, replacement: three('Three.TSL.js') },
      // three's addons import three, which must be its WebGPU build
      { find: /^three(\/webgpu)?$/, replacement: three('Three.WebGPU.js') },
    ],
  },
  // main.ts awaits at the top level
  build: { target: 'es2022' },
})
