module.exports = {
  root: true,
  ignorePatterns: ["jquery-ui-*/", "node_modules/"],
  env: {
    browser: true,
    jquery: true,
  },
  extends: ["eslint:recommended", "prettier"],
  globals: {
    IPython: "readonly",
    MozWebSocket: "readonly",
  },
  rules: {
    indent: ["error", 2, { SwitchCase: 1 }],
    "no-unused-vars": [
      "error",
      {
        argsIgnorePattern: "^_",
      },
    ],
    quotes: ["error", "double", { avoidEscape: true }],
  },
  overrides: [
    {
      files: "js/**/*.js",
      rules: {
        indent: ["error", 4, { SwitchCase: 1 }],
        quotes: ["error", "single", { avoidEscape: true }],
      },
    },
  ],
};
