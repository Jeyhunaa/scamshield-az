import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The app is served by FastAPI under /app (see app/api/main.py).
// In development, API calls are proxied to the FastAPI server on port 8000.
const api = "http://127.0.0.1:8000";

export default defineConfig({
  base: "/app/",
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/assess": api, "/scenarios": api, "/health": api },
  },
});
