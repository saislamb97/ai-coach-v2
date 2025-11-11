// lib/visemeScheduler.ts
// Timing-only scheduler that trusts server visemes.
// - Locks to <audio>.currentTime when present
// - Derives timeline from viseme_times OR frame_ms OR viseme_fps OR duration_ms
// - Catmull–Rom resampling + critically-damped spring for “muscle” feel
// No expression hacks here — all articulation lives on the server.

export type AudioChunkMsg = {
  // Audio (optional)
  audio?: string;                 // base64 audio payload
  audio_format?: string;          // default: "mp3"

  // Visemes (required)
  viseme: number[][];             // [N][15] ARKit-15
  viseme_times?: number[];        // seconds, length == N (preferred)

  // Fallback timing hints (optional)
  duration_ms?: number;           // server calc (fallback only)
  frame_ms?: number;              // e.g., 11 if VIS_FPS=90
  viseme_fps?: number;            // e.g., 90

  // Metadata (optional)
  viseme_format?: string;         // "arkit15"
  viseme_profile?: string;        // e.g., "arkit15-v2" (server-side shaping version)
  chunk_index?: number;
  offset_ms?: number;
};

const COLS = 15;

function clamp01(x: number) { return x < 0 ? 0 : (x > 1 ? 1 : x); }
function lerp(a: number, b: number, t: number) { return a + (b - a) * (t < 0 ? 0 : t > 1 ? 1 : t); }
function catmullRom(p0: number, p1: number, p2: number, p3: number, t: number) {
  const t2 = t * t, t3 = t2 * t;
  return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3);
}
function lowerBound(arr: Float32Array, t: number): number {
  let lo = 0, hi = arr.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >>> 1;
    if (arr[mid] < t) lo = mid + 1; else hi = mid - 1;
  }
  return Math.max(1, Math.min(arr.length - 1, lo));
}

function buildTimes(
  framesN: number,
  msg: AudioChunkMsg
): Float32Array {
  // Preferred: server-provided viseme_times
  if (Array.isArray(msg.viseme_times) && msg.viseme_times.length >= 2) {
    const t = Float32Array.from(msg.viseme_times.map(Number)).slice(0, framesN);
    // Ensure strictly non-decreasing and starts at 0
    const out = new Float32Array(t.length);
    let prev = 0;
    for (let i = 0; i < t.length; i++) { const v = Math.max(0, isFinite(t[i]) ? t[i] : prev); out[i] = v < prev ? prev : v; prev = out[i]; }
    return out;
  }

  // 2) frame_ms
  if (msg.frame_ms && msg.frame_ms > 0) {
    const dt = msg.frame_ms / 1000;
    const out = new Float32Array(framesN);
    for (let i = 0; i < framesN; i++) out[i] = i * dt;
    return out;
  }

  // 3) viseme_fps
  if (msg.viseme_fps && msg.viseme_fps > 0) {
    const dt = 1 / msg.viseme_fps;
    const out = new Float32Array(framesN);
    for (let i = 0; i < framesN; i++) out[i] = i * dt;
    return out;
  }

  // 4) duration_ms fallback
  const durS = Math.max(0.02, (Number(msg.duration_ms) || 0) / 1000);
  const out = new Float32Array(framesN);
  const step = framesN > 1 ? (durS / (framesN - 1)) : durS;
  for (let i = 0; i < framesN; i++) out[i] = i * step;
  return out;
}

export function createVisemeScheduler(opts?: {
  // Spring (muscle) params
  zeta?: number;              // damping ratio (~0.8–1.0)
  freqBase?: number;          // Hz for most shapes
  freqJaw?: number;           // Hz for jawOpen
  exaggeration?: number;      // subtle pop after spring
  // Crossfade & idle relax
  crossfadeS?: number;
  neutralDecayPerSec?: number;
}) {
  type Chunk = {
    frames: Float32Array[];   // [N][15] typed
    timesRaw: Float32Array;   // as sent/derived (seconds)
    times: Float32Array;      // scaled to element duration
    duration: number;         // seconds (authoritative)
    audioEl?: HTMLAudioElement;
    startedAt: number;        // performance.now()/1000
    fadeInEnd: number;        // absolute time (s) for crossfade end
    fadeInFrom: Float32Array; // previous visible pose
    done: boolean;
  };

  const zeta = opts?.zeta ?? 0.84;
  const freqBase = opts?.freqBase ?? 7.5;
  const freqJaw  = opts?.freqJaw  ?? 10.0;
  const exaggeration = opts?.exaggeration ?? 1.04;
  const crossfadeS = opts?.crossfadeS ?? 0.085;
  const neutralDecayPerSec = opts?.neutralDecayPerSec ?? 1.0;

  const queue: Chunk[] = [];
  let active: Chunk | null = null;
  let muted = false;

  // Pose state
  const pos = new Float32Array(COLS);
  const vel = new Float32Array(COLS);
  const tmp = new Float32Array(COLS);

  let lastTick = performance.now() / 1000;

  function startNext() {
    if (active || queue.length === 0) return;
    const ch = queue.shift()!;
    active = ch;

    const begin = (durEl: number) => {
      // Scale timeline to actual element duration (keeps sync)
      const rawTail = ch.timesRaw[ch.timesRaw.length - 1] || 0;
      const scale = rawTail > 0 ? (durEl / rawTail) : 1;
      ch.times = Float32Array.from(ch.timesRaw, (t) => t * scale);
      ch.duration = ch.times[ch.times.length - 1] || Math.max(0.02, durEl);

      ch.startedAt = performance.now() / 1000;
      ch.fadeInFrom = pos.slice(0);
      ch.fadeInEnd = ch.startedAt + crossfadeS;

      if (ch.audioEl && !muted) {
        (ch.audioEl as any).playsInline = true;
        void ch.audioEl.play().catch(()=>{});
        ch.audioEl.onended = () => { ch.done = true; active = null; startNext(); };
      } else {
        // Non-audio fallback timer
        window.setTimeout(() => { ch.done = true; active = null; startNext(); }, Math.round(ch.duration * 1000) + 10);
      }
    };

    if (ch.audioEl && !muted) {
      const onMeta = () => {
        const d = (isFinite(ch.audioEl!.duration) && ch.audioEl!.duration > 0)
          ? ch.audioEl!.duration
          : (ch.duration || (ch.timesRaw[ch.timesRaw.length - 1] || 0) || 0.5);
        ch.audioEl!.removeEventListener('loadedmetadata', onMeta);
        begin(d);
      };
      ch.audioEl.addEventListener('loadedmetadata', onMeta);
      if (!Number.isNaN(ch.audioEl.duration) && ch.audioEl.duration > 0) onMeta();
    } else {
      const d = ch.duration || (ch.timesRaw[ch.timesRaw.length - 1] || 0);
      begin(d);
    }
  }

  function sampleCatmullRom(ch: Chunk, t: number, out: Float32Array) {
    const times = ch.times;
    const idx1 = lowerBound(times, t);
    const i1 = idx1, i0 = idx1 - 1;
    const i2 = Math.min(i1 + 1, ch.frames.length - 1);
    const i_1 = Math.max(i0 - 1, 0);
    const t0 = times[i0], t1 = times[i1];
    const u = (t1 > t0) ? (t - t0) / (t1 - t0) : 0;

    const f_1 = ch.frames[i_1] || ch.frames[0];
    const f0  = ch.frames[i0]  || ch.frames[0];
    const f1  = ch.frames[i1]  || ch.frames[ch.frames.length - 1];
    const f2  = ch.frames[i2]  || ch.frames[ch.frames.length - 1];

    for (let i = 0; i < COLS; i++) {
      out[i] = clamp01(catmullRom(f_1[i], f0[i], f1[i], f2[i], u));
    }
  }

  return {
    pushChunk(msg: AudioChunkMsg) {
      const frames = (msg.viseme || []).map(
        row => Float32Array.from((row || []).slice(0, COLS).map(v => clamp01(Number(v) || 0)))
      );
      const timesRaw = buildTimes(frames.length, msg);
      const ch: Chunk = {
        frames,
        timesRaw,
        times: new Float32Array(0),
        duration: Math.max(0.02, (Number(msg.duration_ms) || 0) / 1000),
        audioEl: undefined,
        startedAt: 0,
        done: false,
        fadeInEnd: 0,
        fadeInFrom: new Float32Array(COLS),
      };

      // audio (optional)
      const fmt = (msg.audio_format || 'mp3').toLowerCase();
      if (msg.audio && !muted) {
        ch.audioEl = new Audio(`data:audio/${fmt};base64,${msg.audio}`);
        ch.audioEl.preload = 'auto';
      }

      queue.push(ch);
      startNext();
    },

    getFrame(): number[] | null {
      const now = performance.now() / 1000;
      const dt = Math.max(0.001, Math.min(0.033, now - lastTick));
      lastTick = now;

      const ch = active;
      if (!ch || !ch.frames.length || !ch.times.length) {
        // Neutral relaxation while idle
        const k = Math.exp(-neutralDecayPerSec * dt);
        for (let i = 0; i < COLS; i++) { pos[i] *= k; vel[i] *= k; }
        return Array.from(pos);
      }

      // Sync to audio element when possible
      let t = now - ch.startedAt;
      if (ch.audioEl && !Number.isNaN(ch.audioEl.currentTime)) {
        t = Math.max(0, Math.min(ch.audioEl.currentTime, ch.duration));
      } else {
        t = Math.min(Math.max(0, t), ch.duration);
      }

      // Target pose via Catmull–Rom
      sampleCatmullRom(ch, t, tmp);

      // Crossfade from previous chunk
      if (now < ch.fadeInEnd) {
        const cfk = 1 - Math.max(0, (ch.fadeInEnd - now) / crossfadeS);
        for (let i = 0; i < COLS; i++) tmp[i] = lerp(ch.fadeInFrom[i], tmp[i], cfk);
      }

      // Critically-damped spring (muscle)
      for (let i = 0; i < COLS; i++) {
        const f = (i === 0) ? freqJaw : freqBase;
        const w = 2 * Math.PI * f;
        const acc = w * w * (tmp[i] - pos[i]) - 2 * zeta * w * vel[i];
        vel[i] += acc * dt;
        pos[i] += vel[i] * dt;
        pos[i] = clamp01(pos[i] * (exaggeration || 1));
      }

      return Array.from(pos);
    },

    setMuted(v: boolean) {
      muted = !!v;
      if (muted && active?.audioEl) { try { active.audioEl.pause(); } catch {} }
    },

    stop() {
      queue.length = 0;
      if (active?.audioEl) { try { active.audioEl.pause(); } catch {} }
      active = null;
    },
  };
}
