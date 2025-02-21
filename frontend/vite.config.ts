import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig({
    plugins: [react()],
    build: {
        outDir: "../static",
        emptyOutDir: true,
        sourcemap: true
    },
    server: {
        proxy: {
            // "/conversation": "http://127.0.0.1:5000",
            "/ask": "http://localhost:5000",
            "/chat": "http://localhost:5000",
            "/history": "http://localhost:5000",
            "/blob-sas-url":"http://localhost:5000"
        }
    }
});
