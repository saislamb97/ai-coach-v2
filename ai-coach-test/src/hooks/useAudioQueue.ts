// hooks/useAudioQueue.ts
import * as React from 'react'

type UseAudioQueueOpts = {
  getVisemeFrameRef: React.MutableRefObject<() => number[] | null>
  muted: boolean
}

export function useAudioQueue({ getVisemeFrameRef, muted }: UseAudioQueueOpts) {
  const audioRef = React.useRef<HTMLAudioElement | null>(
    typeof Audio !== 'undefined' ? new Audio() : null
  )

  // queue + playback state
  const q = React.useRef<Array<{ url: string; frames: number[][]; times: number[] }>>([]).current
  const active = React.useRef<{ url: string | null; frames: number[][]; times: number[] }>({
    url: null,
    frames: [],
    times: [],
  })
  const playing = React.useRef(false)
  const raf = React.useRef<number | null>(null)
  const lastIdx = React.useRef(0)
  const clockStartRef = React.useRef<number | null>(null)

  // ---- helpers ----
  const cancelLoop = React.useCallback(() => {
    if (raf.current) cancelAnimationFrame(raf.current)
  }, [])

  const startLoop = React.useCallback(() => {
    cancelLoop()
    const step = () => {
      const audio = audioRef.current
      let T = 0
      if (audio && !Number.isNaN(audio.currentTime) && audio.currentTime > 0 && !audio.paused) {
        T = audio.currentTime
      } else if (clockStartRef.current != null) {
        T = (performance.now() - clockStartRef.current) / 1000
      }

      const { frames, times } = active.current
      if (!(frames.length && times.length === frames.length)) {
        raf.current = requestAnimationFrame(step)
        return
      }

      if (T <= times[0]) {
        getVisemeFrameRef.current = () => frames[0]
      } else if (T >= times[times.length - 1]) {
        getVisemeFrameRef.current = () => frames[times.length - 1]
      } else {
        while (lastIdx.current < times.length - 2 && T >= times[lastIdx.current + 1]) lastIdx.current++
        const t0 = times[lastIdx.current]
        const t1 = times[lastIdx.current + 1]
        const f0 = frames[lastIdx.current]
        const f1 = frames[lastIdx.current + 1]
        const a = t1 > t0 ? (T - t0) / (t1 - t0) : 0
        const lerp = f0.map((v, i) => v + (f1[i] - v) * a)
        getVisemeFrameRef.current = () => lerp
      }
      raf.current = requestAnimationFrame(step)
    }
    raf.current = requestAnimationFrame(step)
  }, [cancelLoop, getVisemeFrameRef])

  const finishAndNext = React.useCallback(() => {
    cancelLoop()
    if (active.current.url) URL.revokeObjectURL(active.current.url)
    active.current = { url: null, frames: [], times: [] }
    getVisemeFrameRef.current = () => null
    lastIdx.current = 0
    playing.current = false
    clockStartRef.current = null
    void playNext()
  }, [cancelLoop, getVisemeFrameRef])

  const b64ToMp3Blob = React.useCallback((b64: string) => {
    let raw = b64 || ''
    if (raw.startsWith('data:')) raw = raw.split(',')[1] || ''
    raw = raw.replace(/[\r\n\s]/g, '').replace(/-/g, '+').replace(/_/g, '/') // tolerate url-safe
    const pad = raw.length % 4
    if (pad) raw += '='.repeat(4 - pad)
    const bin = atob(raw)
    const buf = new Uint8Array(bin.length)
    for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i)
    return new Blob([buf], { type: 'audio/mpeg' })
  }, [])

  const stopAll = React.useCallback(() => {
    try {
      audioRef.current?.pause()
    } catch {}
    cancelLoop()
    if (active.current.url) URL.revokeObjectURL(active.current.url)
    active.current = { url: null, frames: [], times: [] }
    while (q.length) {
      const it = q.shift()!
      URL.revokeObjectURL(it.url)
    }
    getVisemeFrameRef.current = () => null
    lastIdx.current = 0
    playing.current = false
    clockStartRef.current = null
  }, [cancelLoop, getVisemeFrameRef, q])

  const playNext = React.useCallback(async () => {
    if (playing.current || !q.length) return
    playing.current = true
    const item = q.shift()!
    if (active.current.url) URL.revokeObjectURL(active.current.url)
    active.current = { url: item.url, frames: item.frames, times: item.times }
    const audio = audioRef.current
    clockStartRef.current = performance.now()
    if (audio) {
      audio.src = item.url
      audio.preload = 'auto'
      audio.muted = muted
      audio.onplay = () => startLoop()
      audio.onpause = () => {
        if (!audio.ended) cancelLoop()
      }
      audio.onended = () => finishAndNext()
      audio.onerror = () => finishAndNext()
      try {
        await audio.play()
      } catch {
        // Autoplay could be blocked → still drive visemes by clock
        startLoop()
      }
    }
  }, [cancelLoop, finishAndNext, muted, q, startLoop])

  const enqueue = React.useCallback(
    (b64: string, frames: number[][], times: number[]) => {
      if (!Array.isArray(frames) || !Array.isArray(times) || frames.length !== times.length) return
      const url = URL.createObjectURL(b64ToMp3Blob(b64))
      q.push({ url, frames, times })
      void playNext()
    },
    [b64ToMp3Blob, playNext, q]
  )

  // keep <audio> mute in sync
  React.useEffect(() => {
    if (audioRef.current) audioRef.current.muted = muted
  }, [muted])

  // cleanup
  React.useEffect(() => stopAll, [stopAll])

  return {
    enqueue,   // enqueue(b64, frames, times)
    stopAll,   // stop all audio and clear queue
    audioEl: audioRef.current,
  }
}
