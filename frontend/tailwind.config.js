/** @type {import('tailwindcss').Config} */
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,jsx,ts,tsx}'
  ],
  theme: {
    extend: {
      colors: {
        // Custom brand palette (dark -> light)
        primary: {
          900: '#212529',
          700: '#343a40',
          500: '#495057',
          300: '#6c757d',
          DEFAULT: '#2d6a4f'
        }
      }
    },
  },
  plugins: [],
}

