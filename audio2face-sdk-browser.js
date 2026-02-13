import { Audio2FaceCore } from './audio2face-core.mjs';

const ort = window.ort;
if (!ort) throw new Error('ONNX Runtime not loaded. Include ort.webgpu.min.js or onnxruntime-web.min.js before this script.');

export class Audio2FaceSDK extends Audio2FaceCore {
  constructor(options = {}) {
    super({ ort, ...options });
    this.actualBackend = null;
  }

  static get BLENDSHAPE_NAMES() { return Audio2FaceCore.BLENDSHAPE_NAMES; }
  static get EMOTIONS() { return Audio2FaceCore.EMOTIONS; }

  isWebGPUSupported() {
    return typeof navigator !== 'undefined' && navigator.gpu !== undefined;
  }

  async loadModel(modelFile, options = {}) {
    const { useGPU = true } = options;
    let modelBuffer;
    if (modelFile instanceof File || modelFile instanceof Blob) {
      modelBuffer = await modelFile.arrayBuffer();
    } else if (typeof modelFile === 'string') {
      modelBuffer = await (await fetch(modelFile)).arrayBuffer();
    } else if (modelFile instanceof ArrayBuffer) {
      modelBuffer = modelFile;
    } else {
      throw new Error('Model must be a File, Blob, ArrayBuffer, or URL string');
    }

    const sessionOpts = { graphOptimizationLevel: 'all' };

    if (useGPU && this.isWebGPUSupported()) {
      try {
        sessionOpts.executionProviders = ['webgpu'];
        this.session = await ort.InferenceSession.create(modelBuffer, sessionOpts);
        this.actualBackend = 'webgpu';
        return this.session;
      } catch (err) {
        console.warn('WebGPU failed, falling back to WASM:', err.message);
      }
    }
    sessionOpts.executionProviders = ['wasm'];
    this.session = await ort.InferenceSession.create(modelBuffer, sessionOpts);
    this.actualBackend = 'wasm';
    return this.session;
  }

  async processAudioFile(audioFile) {
    if (!this.session) throw new Error('Model not loaded. Call loadModel() first.');
    const arrayBuffer = await audioFile.arrayBuffer();
    const ctx = new OfflineAudioContext(1, 1, this.sampleRate);
    const decoded = await ctx.decodeAudioData(arrayBuffer);
    let audioData = decoded.getChannelData(0);
    if (decoded.sampleRate !== this.sampleRate) {
      audioData = this.resampleAudio(audioData, decoded.sampleRate, this.sampleRate);
    }
    const results = [];
    for (let i = 0; i < audioData.length - this.bufferLen; i += this.bufferOfs) {
      results.push(await this.runInference(audioData.slice(i, i + this.bufferLen)));
    }
    return this.aggregateResults(results);
  }
}

export default Audio2FaceSDK;
