import { globalIgnores } from "eslint/config";

import commonConfig from "../eslint.config.common.mjs";

/** @type { import("eslint").Linter.Config[] } */
export default [
  ...commonConfig,
  // Linted by its own package.
  globalIgnores(["integration-test/**"]),
];
