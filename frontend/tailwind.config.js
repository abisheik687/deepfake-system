
/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                'cyber-black': '#0a0a0a',
                'cyber-gray': '#1a1a1a',
                'neon-blue': '#00f3ff',
                'neon-red': '#ff003c',
                'neon-green': '#00ff9d',
            },
            fontFamily: {
                'mono': ['"Courier New"', 'monospace'],
                'sans': ['Inter', 'sans-serif'],
            }
        },
    },
    plugins: [],
}
