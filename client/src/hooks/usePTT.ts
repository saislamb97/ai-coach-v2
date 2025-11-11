// hooks/usePTT.ts
import * as React from 'react'

export type MicState = 'idle' | 'recording' | 'blocked' | 'denied' | 'unsupported'

type UsePTTOpts = {
  muted: boolean
  onSend: (payload: { base64Audio: string; format: 'webm' | 'm4a' | 'ogg' }) => void
}

export function usePTT({ onSend }: UsePTTOpts) {
  const [micSupported, setMicSupported] = React.useState<boolean>(true)
  const [micState, setMicState] = React.useState<MicState>('idle')
  const recStream = React.useRef<MediaStream | null>(null)
  const recorder = React.useRef<MediaRecorder | null>(null)
  const chunks = React.useRef<BlobPart[]>([])

  // support & permission probe
  React.useEffect(() => {
    // @ts-ignore
    if (typeof MediaRecorder === 'undefined') {
      setMicSupported(false)
      setMicState('unsupported')
      return
    }
    setMicSupported(true)
    if ((navigator as any).permissions?.query) {
      ;(navigator as any).permissions
        .query({ name: 'microphone' as PermissionName })
        .then((st: any) => {
          setMicState(st.state === 'denied' ? 'denied' : 'idle')
          st.onchange = () => setMicState(st.state === 'denied' ? 'denied' : 'idle')
        })
        .catch(() => {})
    }
  }, [])

  async function requestMicPermission() {
    try {
      const s = await navigator.mediaDevices.getUserMedia({ audio: true })
      s.getTracks().forEach((t) => t.stop())
      setMicState('idle')
      setMicSupported(true)
    } catch {
      setMicState('denied')
    }
  }

  async function ensureMic(): Promise<MediaStream | null> {
    try {
      return await navigator.mediaDevices.getUserMedia({ audio: true })
    } catch {
      return null
    }
  }

  function pickRecorderMime(): string {
    const candidates = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/mp4;codecs=mp4a.40.2',
      'audio/mp4',
      'audio/ogg;codecs=opus',
      'audio/ogg',
    ]
    // @ts-ignore
    if (typeof MediaRecorder === 'undefined') return ''
    for (const m of candidates) {
      // @ts-ignore
      if (MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(m)) return m
    }
    return 'audio/webm'
  }
  function fmtFromMime(m: string): 'webm' | 'm4a' | 'ogg' {
    if (m.includes('mp4')) return 'm4a'
    if (m.includes('ogg')) return 'ogg'
    return 'webm'
  }

  async function startPTT() {
    if (!micSupported) {
      setMicState('unsupported')
      return
    }
    setMicState('recording')

    const stream = await ensureMic()
    if (!stream) {
      setMicState('denied')
      return
    }
    recStream.current = stream

    const mime = pickRecorderMime()
    // @ts-ignore
    const rec = new MediaRecorder(stream, { mimeType: mime, audioBitsPerSecond: 64000 })
    recorder.current = rec
    chunks.current = []

    rec.ondataavailable = (ev: BlobEvent | any) => {
      if (ev.data && ev.data.size) chunks.current.push(ev.data)
    }
    rec.onstop = async () => {
      try {
        const blob = new Blob(chunks.current, { type: mime })
        chunks.current = []
        try {
          stream.getTracks().forEach((t) => t.stop())
        } catch {}
        recStream.current = null

        const base64Audio = await new Promise<string>((resolve, reject) => {
          const fr = new FileReader()
          fr.onloadend = () => resolve((fr.result || '') as string)
          fr.onerror = reject
          fr.readAsDataURL(blob)
        })

        onSend({ base64Audio, format: fmtFromMime(mime) })
      } catch {
        // swallow
      } finally {
        setMicState('idle')
      }
    }

    try {
      rec.start()
    } catch {
      setMicState('blocked')
    }
  }

  async function stopPTT(send = true) {
    if (!recorder.current) {
      setMicState('idle')
      return
    }
    try {
      if (!send) chunks.current = []
      recorder.current.stop()
    } catch {
      setMicState('idle')
    }
  }

  // mouse/touch helpers
  const pttDown = (e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault()
    void startPTT()
  }
  const pttUp = (e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault()
    void stopPTT(true)
  }
  const pttCancel = (e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault()
    void stopPTT(false)
  }

  return {
    micSupported,
    micState,
    requestMicPermission,
    startPTT,
    stopPTT,
    pttDown,
    pttUp,
    pttCancel,
  }
}
