#!/usr/bin/env node
import https from 'https'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const modelPath = path.join(__dirname, '..', 'model.onnx')

if (fs.existsSync(modelPath)) {
  console.log('Model already exists at', modelPath)
  process.exit(0)
}

const MODEL_URL = 'https://github.com/AnEntrypoint/afan/releases/download/v1.0.0/model.onnx'

console.log('Downloading model... (this may take a few minutes)')

const file = fs.createWriteStream(modelPath)
let redirectCount = 0

function download(url) {
  redirectCount++
  if (redirectCount > 10) {
    console.error('Too many redirects')
    process.exit(1)
  }
  
  https.get(url, (res) => {
    if (res.statusCode === 301 || res.statusCode === 302) {
      download(res.headers.location)
      return
    }
    
    if (res.statusCode !== 200) {
      console.error(`Failed to download: ${res.statusCode}`)
      process.exit(1)
    }
    
    const total = parseInt(res.headers['content-length'], 10)
    let downloaded = 0
    
    res.on('data', (chunk) => {
      downloaded += chunk.length
      const pct = total ? Math.round(downloaded / total * 100) : 0
      process.stdout.write(`\rDownloading: ${pct}%`)
    })
    
    res.pipe(file)
    
    file.on('finish', () => {
      file.close()
      console.log('\nModel downloaded successfully!')
    })
  }).on('error', (err) => {
    console.error('Download failed:', err.message)
    process.exit(1)
  })
}

download(MODEL_URL)