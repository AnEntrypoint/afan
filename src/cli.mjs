#!/usr/bin/env node
import { createAudio2FaceAPI } from './api.mjs'
import path from 'path'
import { fileURLToPath } from 'url'
import fs from 'fs'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

const args = process.argv.slice(2)
const port = parseInt(args.find(a => a.startsWith('--port='))?.split('=')[1]) || 3000
const host = args.find(a => a.startsWith('--host='))?.split('=')[1] || '0.0.0.0'
const help = args.includes('--help') || args.includes('-h')

if (help) {
  console.log(`
afan - Audio-driven facial animation server

Usage: afan [options]

Options:
  --port=PORT    Server port (default: 3000)
  --host=HOST    Server host (default: 0.0.0.0)
  --help, -h     Show this help

API Endpoints:
  GET  /health       Server status
  POST /process      Audio → animation (multipart, raw, or JSON)
  GET  /config       SDK configuration info

Examples:
  afan
  afan --port=8080
  curl -X POST --data-binary audio.raw http://localhost:3000/process > out.afan
`)
  process.exit(0)
}

const modelPath = path.join(__dirname, '..', 'model.onnx')
const configPath = path.join(__dirname, '..', 'config.json')

if (!fs.existsSync(modelPath)) {
  console.error(`Model not found: ${modelPath}`)
  console.error('Run: afan-download to download the model')
  process.exit(1)
}

const server = createAudio2FaceAPI({
  modelPath,
  configPath,
  fps: 30,
  useGPU: false
})

server.app.get('/', (req, res) => {
  res.json({
    name: 'afan',
    version: '1.0.0',
    endpoints: {
      'GET /health': 'Server status',
      'GET /config': 'SDK configuration',  
      'POST /process': 'Audio → animation'
    }
  })
})

server.start(port).then(() => {
  console.log(`afan server running on http://${host}:${port}`)
})