/** @type {import('tailwindcss').Config} */
// Tailwind scans these UI folders when deciding which utility classes to generate.
module.exports = {
  content: [
    "./pages/**/*.{js,jsx,mjs}",
    "./components/**/*.{js,jsx,mjs}"
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
