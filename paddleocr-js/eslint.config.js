import eslint from "@eslint/js";
import tseslint from "typescript-eslint";
import globals from "globals";

export default tseslint.config(
  {
    ignores: ["**/dist", "**/node_modules", "**/coverage", "**/.cache"],
  },
  eslint.configs.recommended,
  {
    files: ["packages/**/*.ts"],
    extends: [...tseslint.configs.strictTypeChecked],
    languageOptions: {
      globals: { ...globals.browser },
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
  },
  {
    files: ["apps/**/*.js", "*.config.{js,ts}", "packages/**/*.config.*"],
    languageOptions: {
      globals: { ...globals.browser, ...globals.node },
    },
  },
);
