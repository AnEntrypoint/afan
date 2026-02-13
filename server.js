import express from 'express';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { Audio2FaceSDK } from './audio2face-sdk-node.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const app = express();
const PORT = process.env.PORT || 8765;

app.use(express.json({ limit: '50mb' }));
app.use(express.raw({ type: 'audio/*', limit: '50mb' }));
app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.sendStatus(204);
  next();
});

const sdk = new Audio2FaceSDK();

async function loadModel() {
  try {
    const configPath = path.join(__dirname, 'a2f_ms_config.json');
    if (fs.existsSync(configPath)) {
      await sdk.loadConfigFile(configPath);
      console.log('Loaded model config from a2f_ms_config.json');
    }
    const modelPath = path.join(__dirname, 'network_actual.onnx');
    if (!fs.existsSync(modelPath)) {
      console.error('Model file not found:', modelPath);
      return false;
    }
    console.log('Loading model...');
    await sdk.loadModel(modelPath, { useGPU: false });
    console.log(`Model loaded (${sdk.actualBackend})`);
    console.log(`  Inputs: ${sdk.session.inputNames.join(', ')}`);
    console.log(`  Outputs: ${sdk.session.outputNames.join(', ')}`);
    return true;
  } catch (err) {
    console.error('Failed to load model:', err.message);
    return false;
  }
}

function requireModel(req, res, next) {
  if (!sdk.session) return res.status(503).json({ error: 'MODEL_NOT_LOADED', message: 'Model is still loading. Try again shortly.' });
  next();
}

app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', modelLoaded: !!sdk.session, backend: sdk.actualBackend, timestamp: Date.now() });
});

app.get('/api/config', (req, res) => {
  res.json({
    emotions: Audio2FaceSDK.EMOTIONS,
    blendshapes: Audio2FaceSDK.BLENDSHAPE_NAMES,
    sampleRate: sdk.sampleRate,
    bufferLen: sdk.bufferLen,
    bufferOfs: sdk.bufferOfs,
    faceParams: sdk.faceParams
  });
});

app.post('/api/emotion', requireModel, (req, res) => {
  try {
    sdk.setEmotions(req.body);
    res.json({ ok: true, emotions: Object.fromEntries(Audio2FaceSDK.EMOTIONS.map((e, i) => [e, sdk.emotionVector[i]])) });
  } catch (err) {
    res.status(400).json({ error: 'INVALID_EMOTION', message: err.message });
  }
});

app.post('/api/process', requireModel, async (req, res) => {
  try {
    const audioBuffer = req.body;
    const audioData = new Float32Array(audioBuffer.buffer, audioBuffer.byteOffset, audioBuffer.byteLength / 4);
    res.json(await sdk.processAudioFile(audioData));
  } catch (err) {
    console.error('Processing error:', err);
    res.status(500).json({ error: 'PROCESSING_ERROR', message: err.message });
  }
});

app.post('/api/process-chunk', requireModel, async (req, res) => {
  try {
    const { audio, emotion } = req.is('json') ? req.body : { audio: null };
    let audioData;
    if (audio) {
      audioData = new Float32Array(audio);
    } else {
      const buf = req.body;
      audioData = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
    }
    res.json(await sdk.processAudioChunk(audioData, emotion ? { emotion } : {}));
  } catch (err) {
    console.error('Processing error:', err);
    res.status(500).json({ error: 'PROCESSING_ERROR', message: err.message });
  }
});

app.use(express.static(__dirname, {
  setHeaders: (res, filePath) => {
    if (filePath.endsWith('.mjs')) res.setHeader('Content-Type', 'application/javascript');
    else if (filePath.endsWith('.wasm')) res.setHeader('Content-Type', 'application/wasm');
  }
}));

app.get('/', (req, res) => { res.sendFile(path.join(__dirname, 'index.html')); });

async function start() {
  await loadModel();
  app.listen(PORT, () => {
    console.log(`\nServer: http://localhost:${PORT}`);
    console.log('Endpoints: /api/health, /api/config, /api/emotion, /api/process, /api/process-chunk\n');
  });
}

start().catch(err => { console.error('Server failed:', err); process.exit(1); });
process.on('SIGINT', () => { sdk.dispose(); process.exit(0); });
