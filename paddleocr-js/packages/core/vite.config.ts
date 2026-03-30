import { resolve } from 'node:path'
import { defineConfig } from 'vite'
import dts from 'vite-plugin-dts'

export default defineConfig({
  plugins: [
    dts({
      rollupTypes: true,
    }),
  ],
  build: {
    lib: {
      entry: resolve(__dirname, 'src/index.ts'),
      name: 'paddleocr',
      formats: ['es', 'cjs', 'umd'],
      fileName: (format) => {
        if (format === 'es') return 'index.mjs'
        if (format === 'cjs') return 'index.cjs'
        return 'index.umd.js'
      },
    },
    rollupOptions: {
      external: [
        'onnxruntime-web',
        '@techstark/opencv-js',
        'clipper-lib',
        'js-yaml',
      ],
      output: {
        globals: {
          'onnxruntime-web': 'ort',
          '@techstark/opencv-js': 'cv',
          'clipper-lib': 'ClipperLib',
          'js-yaml': 'jsyaml',
        },
      },
    },
    sourcemap: true,
    minify: false,
  },
})
