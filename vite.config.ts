import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Relative, so the build works from any path (GitHub Pages serves it
  // under /<repo>/)
  base: './',
  resolve: {
    // Everything, including R3F and three's addons, imports three's
    // WebGPU build, so the unused WebGL renderer and shaders stay out
    alias: [{ find: /^three$/, replacement: 'three/webgpu' }],
  },
  build: {
    rollupOptions: {
      onwarn(warning, warn) {
        // R3F's fallback WebGLRenderer, unused: the app passes its own
        if (warning.message.includes('"WebGLRenderer" is not exported')) return
        warn(warning)
      },
    },
  },
})
