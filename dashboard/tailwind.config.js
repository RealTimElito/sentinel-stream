/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
    "./public/index.html"
  ],
  theme: {
    extend: {
      colors: {
        'sentinel-red': '#ef4444',
        'sentinel-dark': '#1e293b',
        'sentinel-blue': '#3b82f6',
      },
    },
  },
  plugins: [],
}

