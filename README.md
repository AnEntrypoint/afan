# afan

Audio-driven facial animation for VRM. Convert audio to ARKit blendshapes.

## Install

```bash
npm install afan
```

## CLI

```bash
bunx afan
# Server running on http://localhost:3000

# Custom port
bunx afan --port=8080
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server status |
| `/process` | POST | Audio → animation |

### Process Audio

```bash
# Raw Float32 PCM audio (16kHz mono)
curl -X POST --data-binary audio.raw http://localhost:3000/process > out.afan

# Get JSON output
curl -X POST --data-binary audio.raw 'http://localhost:3000/process?format=json'

# Base64 encoded
curl -X POST -H 'Content-Type: application/json' \
  -d '{"data":"base64-audio..."}' \
  'http://localhost:3000/process?format=base64'
```

## Library

```js
import { Audio2FaceSDK } from 'afan'
import { AnimationWriter, AnimationReader } from 'afan/animation'
import { createAudio2FaceAPI } from 'afan/api'

// SDK usage
const sdk = new Audio2FaceSDK()
await sdk.loadConfigFile('./config.json')
await sdk.loadModel('./model.onnx')
const result = await sdk.processAudioChunk(audioFloat32)

// API server
const server = createAudio2FaceAPI({ fps: 30 })
server.start(3000)
```

## Animation Format (.afan)

Compact binary format for facial animation:

```
Header (12 bytes):
  - Magic: 0x4146414E ("NAFA")
  - Version, FPS, numBlendshapes
  - Frame count

Frame data:
  - 52 bytes per frame (1 byte per ARKit blendshape)
```

Size: ~1.7KB per second at 30fps (vs 23KB for JSON).

## License

MIT