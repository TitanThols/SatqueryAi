import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/analyze': 'http://localhost:8000',
      '/tools':   'http://localhost:8000',
      '/health':  'http://localhost:8000',
      '/results': 'http://localhost:8000',
      '/report':  'http://localhost:8000',
    }
  }
})
