import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    coverage: {
      provider: "v8",
      include: ["packages/core/src/**/*.js"],
      exclude: [
        "packages/core/test/**",
        "packages/core/src/**/*.test.js",
        "packages/core/src/**/*.spec.js"
      ],
      reportOnFailure: true,
      reporter: ["text", "html"]
    }
  }
});
