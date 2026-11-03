import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Relative, so the build works from any path (GitHub Pages serves it
  // under /<repo>/)
  base: './',
})
