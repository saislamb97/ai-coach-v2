import { defineConfig, loadEnv } from "vite";
import path from "path";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");

  const DEV_PORT = Number(env.VITE_DEV_PORT || 3000);
  const DEV_HOST = env.VITE_DEV_HOST || true; // true = listen on all
  const PROXY_API = env.VITE_PROXY_API || "/api";
  const PROXY_WS  = env.VITE_PROXY_WS  || "/ws";

  const API_TARGET = (env.VITE_API_BASE || "http://127.0.0.1:8000").replace(/\/+$/, "");
  // Vite's ws proxy can take ws:// or http://, but we’ll pass ws:// explicitly:
  const WS_TARGET  = (env.VITE_WS_TARGET || "ws://127.0.0.1:8000").replace(/\/+$/, "");

  return {
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: { "@": path.resolve(__dirname, "src") },
    },
    server: {
      port: DEV_PORT,
      host: DEV_HOST,
      open: true,
      cors: true,
      proxy: {
        [PROXY_API]: {
          target: API_TARGET,
          changeOrigin: true,
        },
        [PROXY_WS]: {
          target: WS_TARGET,
          ws: true,
          changeOrigin: true,
        },
      },
    },
    build: {
      outDir: path.resolve(__dirname, "../app/assets/client"),
      emptyOutDir: true,
      chunkSizeWarningLimit: 5000,
      rollupOptions: {
        input: { main: path.resolve(__dirname, "index.html") },
        output: {
          entryFileNames: "aicoach.js",
          chunkFileNames: "aicoach-[name].js",
          assetFileNames: (assetInfo) => {
            const ext = path.extname(assetInfo.name ?? "");
            return `aicoach${ext}`;
          },
        },
      },
    },
    define: {
      __APP_VERSION__: JSON.stringify(process.env.npm_package_version),
    },
  };
});
