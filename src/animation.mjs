const ARKIT_NAMES = [
  'browInnerUp', 'browDownLeft', 'browDownRight', 'browOuterUpLeft', 'browOuterUpRight',
  'eyeLookUpLeft', 'eyeLookUpRight', 'eyeLookDownLeft', 'eyeLookDownRight',
  'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight',
  'eyeBlinkLeft', 'eyeBlinkRight', 'eyeSquintLeft', 'eyeSquintRight',
  'eyeWideLeft', 'eyeWideRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight',
  'noseSneerLeft', 'noseSneerRight', 'jawOpen', 'jawForward', 'jawLeft', 'jawRight',
  'mouthFunnel', 'mouthPucker', 'mouthLeft', 'mouthRight',
  'mouthRollUpper', 'mouthRollLower', 'mouthShrugUpper', 'mouthShrugLower',
  'mouthOpen', 'mouthClose', 'mouthSmileLeft', 'mouthSmileRight',
  'mouthFrownLeft', 'mouthFrownRight', 'mouthDimpleLeft', 'mouthDimpleRight',
  'mouthUpperUpLeft', 'mouthUpperUpRight', 'mouthLowerDownLeft', 'mouthLowerDownRight',
  'mouthPressLeft', 'mouthPressRight', 'mouthStretchLeft', 'mouthStretchRight'
]

const MAGIC = 0x4146414E
const VERSION = 2

export class AnimationWriter {
  constructor(options = {}) {
    this.fps = options.fps || 30
    this.sampleRate = options.sampleRate || 16000
    this.samplesPerFrame = Math.floor(this.sampleRate / this.fps)
    this.numBlendshapes = ARKIT_NAMES.length
    this.frameData = []
    this.currentFrameValues = new Uint8Array(this.numBlendshapes)
    this.currentFrameSamples = 0
    this.frameIndex = 0
    this.nameToIndex = {}
    for (let i = 0; i < ARKIT_NAMES.length; i++) {
      this.nameToIndex[ARKIT_NAMES[i]] = i
    }
  }

  processResult(result, samplesInChunk = 0) {
    if (result.blendshapes) {
      for (const bs of result.blendshapes) {
        const idx = this.nameToIndex[bs.name]
        if (idx !== undefined) {
          this.currentFrameValues[idx] = Math.round(bs.value * 255)
        }
      }
    }
    this.currentFrameSamples += samplesInChunk
    
    if (this.currentFrameSamples >= this.samplesPerFrame) {
      this.frameData.push(new Uint8Array(this.currentFrameValues))
      this.currentFrameValues.fill(0)
      this.currentFrameSamples -= this.samplesPerFrame
      this.frameIndex++
    }
  }

  finalize() {
    if (this.currentFrameSamples > 0) {
      this.frameData.push(new Uint8Array(this.currentFrameValues))
    }
  }

  toBuffer() {
    const numFrames = this.frameData.length
    const headerSize = 12
    const frameSize = this.numBlendshapes
    const totalSize = headerSize + (numFrames * frameSize)
    
    const buf = Buffer.alloc(totalSize)
    let offset = 0
    
    buf.writeUInt32LE(MAGIC, offset); offset += 4
    buf.writeUInt8(VERSION, offset); offset += 1
    buf.writeUInt8(this.fps, offset); offset += 1
    buf.writeUInt8(this.numBlendshapes, offset); offset += 1
    buf.writeUInt8(0, offset); offset += 1
    buf.writeUInt32LE(numFrames, offset); offset += 4
    
    for (const frame of this.frameData) {
      for (let i = 0; i < frame.length; i++) {
        buf[offset++] = frame[i]
      }
    }
    
    return buf
  }

  toBase64() {
    return this.toBuffer().toString('base64')
  }

  getFrameCount() {
    return this.frameData.length
  }

  getDuration() {
    return this.frameData.length / this.fps
  }
}

export class AnimationReader {
  constructor() {
    this.fps = 0
    this.numBlendshapes = 0
    this.numFrames = 0
    this.names = ARKIT_NAMES
    this.frames = []
  }

  fromBuffer(buf) {
    let offset = 0
    
    const magic = buf.readUInt32LE(offset); offset += 4
    if (magic !== MAGIC) throw new Error('Invalid animation file')
    
    const version = buf.readUInt8(offset); offset += 1
    if (version < 1 || version > 2) throw new Error(`Unsupported version: ${version}`)
    
    this.fps = buf.readUInt8(offset); offset += 1
    this.numBlendshapes = buf.readUInt8(offset); offset += 1
    offset += 1
    this.numFrames = buf.readUInt32LE(offset); offset += 4
    
    if (version === 1) {
      this.names = []
      for (let i = 0; i < this.numBlendshapes; i++) {
        const len = buf.readUInt8(offset++)
        this.names.push(buf.toString('utf8', offset, offset + len))
        offset += len
      }
    }
    
    this.frames = []
    for (let f = 0; f < this.numFrames; f++) {
      const frame = {}
      for (let i = 0; i < this.numBlendshapes; i++) {
        frame[this.names[i]] = buf[offset++] / 255
      }
      this.frames.push({
        time: f / this.fps,
        blendshapes: frame
      })
    }
    
    return this
  }

  fromBase64(str) {
    return this.fromBuffer(Buffer.from(str, 'base64'))
  }

  getFrame(index) {
    return this.frames[index] || null
  }

  getBlendshapeAt(frameIndex, name) {
    const frame = this.frames[frameIndex]
    return frame ? frame.blendshapes[name] : 0
  }
}

export { ARKIT_NAMES }
export default AnimationWriter