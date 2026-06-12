import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    port: process.env.PORT ? Number(process.env.PORT) : undefined,
    strictPort: !!process.env.PORT,
  },
  build: {
    target: 'esnext',
  },
  resolve: {
    alias: [{ find: /^three$/, replacement: 'three/webgpu' }],
  },
});
