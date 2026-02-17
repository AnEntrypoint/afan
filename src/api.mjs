import express from 'express'
import multer from 'multer'
import { Audio2FaceSDK } from './index.mjs'
import { AnimationWriter } from './animation.mjs'
import path from 'path'
import { fileURLToPath } from 'url'
import fs from 'fs'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

export function createAudio2FaceAPI(options = {}) {
  const app = express()
  const sdk = new Audio2FaceSDK()
  let loaded = false
  
  const modelPath = options.modelPath || path.join(__dirname, '..', 'model.onnx')
  const configPath = options.configPath || path.join(__dirname, '..', 'config.json')
  const fps = options.fps || 30
  const useGPU = options.useGPU ?? false
  
  app.use(express.json({ limit: '50mb' }))
  app.use(express.raw({ type: 'audio/*', limit: '50mb' }))
  app.use(express.raw({ type: 'application/octet-stream', limit: '50mb' }))
  
  const upload = multer({ 
    storage: multer.memoryStorage(),
    limits: { fileSize: 50 * 1024 * 1024 }
  })

  async function loadModel() {
    if (loaded) return
    if (fs.existsSync(configPath)) {
      await sdk.loadConfigFile(configPath)
    }
    await sdk.loadModel(modelPath, { useGPU })
    loaded = true
    console.log(`[a2f] Model loaded (${sdk.actualBackend})`)
  }

  function rawAudioToFloat32(buffer, sampleRate = 16000) {
    const view = new DataView(buffer.buffer || buffer)
    const float32 = new Float32Array(Math.floor(buffer.byteLength / 2))
    for (let i = 0; i < float32.length; i++) {
      float32[i] = view.getInt16(i * 2, true) / 32768
    }
    return float32
  }

  async function processAudio(audioData, inputSampleRate = 16000) {
    await loadModel()
    
    let audio = audioData
    if (audioData instanceof Buffer) {
      audio = rawAudioToFloat32(audioData)
    }

    const writer = new AnimationWriter({ fps, sampleRate: sdk.sampleRate })
    const bufferLen = sdk.bufferLen
    const bufferOfs = sdk.bufferOfs
    let lastBs = null
    let samplesProcessed = 0

    for (let i = 0; i < audio.length - bufferLen; i += bufferOfs) {
      const chunk = audio.slice(i, i + bufferLen)
      let result = await sdk.runInference(chunk)
      
      if (result.blendshapes && lastBs) {
        result.blendshapes = sdk.smoothBlendshapes(lastBs, result.blendshapes)
      }
      lastBs = result.blendshapes
      
      writer.processResult(result, bufferOfs)
      samplesProcessed += bufferOfs
    }
    
    writer.finalize()
    return writer.toBuffer()
  }

  app.get('/health', (req, res) => {
    res.json({ 
      status: 'ok', 
      loaded,
      sampleRate: sdk.sampleRate,
      fps 
    })
  })

  app.post('/process', upload.single('audio'), async (req, res) => {
    try {
      let audioData
      let sampleRate = 16000

      if (req.file) {
        audioData = req.file.buffer
        if (req.body.sampleRate) sampleRate = parseInt(req.body.sampleRate)
      } else if (Buffer.isBuffer(req.body) && req.body.length > 0) {
        audioData = req.body
      } else if (req.body && typeof req.body === 'object') {
        if (req.body.audio) {
          if (typeof req.body.audio === 'string') {
            audioData = Buffer.from(req.body.audio, 'base64')
          } else {
            audioData = Buffer.from(req.body.audio)
          }
          sampleRate = req.body.sampleRate || 16000
        } else if (req.body.data) {
          audioData = Buffer.from(req.body.data, 'base64')
          sampleRate = req.body.sampleRate || 16000
        }
      }
      
      if (!audioData) {
        return res.status(400).json({ error: 'No audio data provided' })
      }

      const animBuffer = await processAudio(audioData, sampleRate)
      
      const format = req.query.format || 'binary'
      
      if (format === 'base64') {
        res.json({
          success: true,
          fps,
          duration: animBuffer.byteLength / (52 + 16) / fps,
          data: animBuffer.toString('base64')
        })
      } else if (format === 'json') {
        const { AnimationReader } = await import('./animation.mjs')
        const reader = new AnimationReader().fromBuffer(animBuffer)
        res.json({
          success: true,
          fps: reader.fps,
          duration: reader.frames.length / reader.fps,
          frames: reader.frames
        })
      } else {
        res.setHeader('Content-Type', 'application/octet-stream')
        res.setHeader('Content-Disposition', 'attachment; filename="animation.afan"')
        res.send(animBuffer)
      }
    } catch (err) {
      console.error('[a2f] Error:', err)
      res.status(500).json({ error: err.message })
    }
  })

  app.post('/process-raw', express.raw({ type: 'application/octet-stream', limit: '50mb' }), async (req, res) => {
    try {
      const audioData = Buffer.from(req.body)
      const animBuffer = await processAudio(audioData)
      
      res.setHeader('Content-Type', 'application/octet-stream')
      res.send(animBuffer)
    } catch (err) {
      res.status(500).json({ error: err.message })
    }
  })

  const server = { app, sdk, loadModel }

  server.start = (port = 3000) => {
    return new Promise((resolve) => {
      app.listen(port, () => {
        console.log(`[a2f] API server running on port ${port}`)
        resolve(server)
      })
    })
  }

  return server
}

export default createAudio2FaceAPI