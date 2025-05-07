/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        navy: {
          50: '#f0f4f8',
          100: '#d9e2ed',
          200: '#b3c6db',
          300: '#84a0c0',
          400: '#5a7ca5',
          500: '#3c5d85',
          600: '#2d4563',
          700: '#233548',
          800: '#1a2533',
          900: '#111927',
        }
      },
      boxShadow: {
        'carrick': '0 4px 14px 0 rgba(35, 53, 72, 0.15)',
      },
      borderColor: {
        DEFAULT: '#d9e2ed', // navy-100
      },
    },
  },
  plugins: []
};
