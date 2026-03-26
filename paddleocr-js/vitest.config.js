import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    coverage: {
      provider: "v8",
      include: ["packages/paddleocr-js/src/**/*.js"],
      exclude: [
        "packages/paddleocr-js/test/**",
        "packages/paddleocr-js/src/**/*.test.js",
        "packages/paddleocr-js/src/**/*.spec.js"
      ],
      reportOnFailure: true,
      reporter: ["text", "html"]
    }
  }
});
