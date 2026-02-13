export interface BlendshapeResult {
  name: string;
  value: number;
}

export interface EyeResult {
  leftX: number;
  leftY: number;
  rightX: number;
  rightY: number;
}

export interface ProcessResult {
  blendshapes: BlendshapeResult[];
  jaw: number;
  eyes: EyeResult | null;
  timestamp: number;
  frameCount?: number;
}

export interface A2FConfig {
  audio_params?: { buffer_len?: number; buffer_ofs?: number; samplerate?: number };
  face_params?: {
    upper_face_smoothing?: number; lower_face_smoothing?: number;
    upper_face_strength?: number; lower_face_strength?: number;
    skin_strength?: number; blink_strength?: number; tongue_strength?: number;
    prediction_delay?: number; emotion?: number[];
    [key: string]: unknown;
  };
  network_params?: {
    num_dim_fullface?: number; num_shapes_skin?: number; num_shapes_tongue?: number;
    result_jaw_size?: number; result_eyes_size?: number;
    explicit_emotions?: string[];
    [key: string]: unknown;
  };
  blendshape_params?: {
    bsWeightMultipliers?: number[]; bsWeightOffsets?: number[];
    bsSolveActivePoses?: number[];
    [key: string]: unknown;
  };
}

export interface SDKOptions {
  sampleRate?: number;
  smoothingFactor?: number;
  config?: A2FConfig;
}

export interface LoadModelOptions {
  useGPU?: boolean;
}

export type EmotionName = 'amazement' | 'anger' | 'cheekiness' | 'disgust' | 'fear' | 'grief' | 'joy' | 'outofbreath' | 'pain' | 'sadness';

export declare class Audio2FaceSDK {
  static readonly BLENDSHAPE_NAMES: string[];
  static readonly EMOTIONS: EmotionName[];

  session: unknown | null;
  actualBackend: string | null;
  sampleRate: number;
  faceParams: Record<string, unknown>;
  emotionVector: Float32Array;

  constructor(options?: SDKOptions);

  loadModel(model: string | ArrayBuffer | Blob | File | Buffer, options?: LoadModelOptions): Promise<unknown>;
  loadConfig(config: A2FConfig): void;
  loadConfigFile?(configPath: string): Promise<A2FConfig>;

  processAudioFile(audio: File | Blob | Float32Array | ArrayBuffer | Buffer): Promise<ProcessResult>;
  processAudioChunk(audio: Float32Array, options?: { emotion?: Partial<Record<EmotionName, number>> }): Promise<ProcessResult>;

  setEmotion(name: EmotionName, value: number): void;
  setEmotions(emotions: Partial<Record<EmotionName, number>>): void;
  getEmotionVector(): Float32Array;

  setSmoothing(factor: number): void;
  setSmoothingRegion(region: 'upper' | 'lower', factor: number): void;

  resampleAudio(audioData: Float32Array, fromRate: number, toRate: number): Float32Array;
  dispose(): void;
}

export default Audio2FaceSDK;
