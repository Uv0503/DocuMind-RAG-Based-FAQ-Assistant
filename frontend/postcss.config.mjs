// const config = {
//   plugins: ["@tailwindcss/postcss"],
// };

// export default config;

// PostCSS delegates Tailwind-specific stylesheet processing to the official plugin.
const config = {
  plugins: {
    "@tailwindcss/postcss": {},
  },
};
export default config;
