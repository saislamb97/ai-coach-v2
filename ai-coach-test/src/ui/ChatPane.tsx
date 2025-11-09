// components/ChatPane.tsx
import React from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkBreaks from 'remark-breaks'
import {
  AiOutlineAudio,
  AiOutlineAudioMuted,
  AiOutlineSend,
  AiOutlineRedo,
  AiOutlineUnlock,
  AiOutlineStop,
} from 'react-icons/ai'
import { FiMic } from 'react-icons/fi'
import { openWS } from '../lib/ws'
import type { WSMessage } from '../lib/ws'
import { rotateSession, getThreadId } from '../lib/session'
import { get } from '../lib/http'
import { createVisemeScheduler } from '../lib/visemeScheduler' // NEW
import { usePTT } from '../hooks/usePTT'

// ---------------- Types ----------------
type EmotionKey = 'joy'|'anger'|'sadness'|'surprise'
type Msg = {
  id: string
  role: 'assistant'|'user'
  text: string
  emotion?: { name: EmotionKey; intensity: number }
  ts?: string
}
type ChatRow = { query?: string; response?: string; created_at?: string; emotion?: any }

export type ChatPaneProps = {
  onSlides: (slides:any)=>void
  onSlidesDone?: () => void
  onSlidesStreaming?: (on: boolean) => void
  getVisemeFrameRef: React.MutableRefObject<() => number[] | null>
  onThreadRotated: (threadId: string)=>void
}

// ---------------- Emotion styling ----------------
const EMOJI: Record<EmotionKey, {emoji:string; dot:string; pill:string; text:string}> = {
  joy:      { emoji:'😊', dot:'bg-amber-400',   pill:'bg-amber-100',   text:'text-amber-800' },
  anger:    { emoji:'😡', dot:'bg-rose-500',    pill:'bg-rose-100',    text:'text-rose-800' },
  sadness:  { emoji:'😢', dot:'bg-blue-500',    pill:'bg-blue-100',    text:'text-blue-800' },
  surprise: { emoji:'😮', dot:'bg-emerald-500', pill:'bg-emerald-100', text:'text-emerald-800' },
}
const neutralEmo = { emoji:'🙂', dot:'bg-slate-400', pill:'bg-slate-200', text:'text-slate-700' }
const uuid = () => (crypto?.randomUUID?.() ?? Math.random().toString(36).slice(2))

// Canonicalize emotion labels
const EMO_ALIAS: Record<string, EmotionKey> = {
  happy:'joy', delight:'joy', pleasure:'joy', love:'joy', content:'joy', joy:'joy',
  mad:'anger', annoyed:'anger', rage:'anger', angry:'anger', anger:'anger',
  sad:'sadness', down:'sadness', blue:'sadness', sorrow:'sadness', sadness:'sadness',
  shocked:'surprise', amazed:'surprise', wow:'surprise', surprise:'surprise'
}
function canonicalEmotionName(s: any): EmotionKey | null {
  const k = String(s ?? '').toLowerCase().trim()
  if (!k) return null
  return EMO_ALIAS[k] ?? ((k in EMOJI) ? (k as EmotionKey) : null)
}
function coerceEmotion(raw: any): { name: EmotionKey; intensity: number } | undefined {
  if (!raw) return undefined
  try {
    const e = typeof raw === 'string' ? JSON.parse(raw) : raw
    const rawName = (e.name || e.emotion || e.label || '').toString()
    const name = canonicalEmotionName(rawName) || 'joy'
    const intensity = Number(e.intensity ?? e.score ?? e.strength ?? 1)
    return { name, intensity: Number.isFinite(intensity) ? intensity : 1 }
  } catch { return undefined }
}

// ============================================================================
// Component
// ============================================================================
export default function ChatPane({
  onSlides, onSlidesDone, onSlidesStreaming, getVisemeFrameRef, onThreadRotated
}: ChatPaneProps){

  // ---- WS ----
  const wsRef = React.useRef<WebSocket|null>(null)
  const [connected, setConnected] = React.useState(false)
  const [muted, setMuted] = React.useState(false)

  // ---- chat + stream ----
  const [msgs, setMsgs] = React.useState<Msg[]>([])
  const [pendingAssistant, setPendingAssistant] = React.useState('')
  const pendingRef = React.useRef('') // authoritative stream buffer
  const [awaitingUserEcho, setAwaitingUserEcho] = React.useState(false)

  // Emotion during stream + header badge
  const [liveEmotion, setLiveEmotion] = React.useState<{name: EmotionKey; intensity: number} | null>(null)
  const [lastEmotion, setLastEmotion] = React.useState<{name: EmotionKey; intensity: number} | null>(null)
  const runEmotionRef = React.useRef<{name: EmotionKey; intensity: number} | null>(null) // per-run

  // Streaming state
  const [streaming, setStreaming] = React.useState(false)
  const [gotFirstToken, setGotFirstToken] = React.useState(false)

  // ------ NEW: Viseme scheduler ------
  const sched = React.useMemo(() => createVisemeScheduler(), [])
  React.useEffect(() => { getVisemeFrameRef.current = sched.getFrame }, [getVisemeFrameRef, sched])

  // ------ slides streaming flag ------
  const slidesStreamingRef = React.useRef(false)

  // ------ autoscroll ------
  const scrollRef = React.useRef<HTMLDivElement>(null)
  const [atBottom, setAtBottom] = React.useState(true)
  React.useEffect(() => {
    if (atBottom) scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight })
  }, [msgs, pendingAssistant, awaitingUserEcho, liveEmotion, streaming, atBottom])
  const onScroll = React.useCallback(() => {
    const el = scrollRef.current
    if (!el) return
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 24
    setAtBottom(nearBottom)
  }, [])

  // preload history (with persisted emotions)
  React.useEffect(() => {
    let mounted = true
    ;(async () => {
      const thread_id = getThreadId()
      if (!thread_id) return
      try {
        const qs = new URLSearchParams({ thread_id, ordering: 'created_at,id', page_size: '200' }).toString()
        const res = await get<{results?: ChatRow[]} | ChatRow[] | ChatRow>(`/api/chats/?${qs}`)
        const list: ChatRow[] =
          Array.isArray((res as any)?.results) ? (res as any).results :
          Array.isArray(res) ? (res as any) :
          res ? [res as any] : []
        const seeded: Msg[] = []
        for (const row of list) {
          if (row.query)     seeded.push({ id: uuid(), role:'user',      text: row.query, ts: row.created_at })
          if (row.response)  seeded.push({ id: uuid(), role:'assistant', text: row.response, ts: row.created_at, emotion: coerceEmotion(row.emotion) as any })
        }
        if (mounted && seeded.length) setMsgs(seeded)
      } catch {}
    })()
    return () => { mounted = false }
  }, [])

  // ------ send text ------
  const [input, setInput] = React.useState('')
  const pendingInputRef = React.useRef('')
  React.useEffect(()=>{ pendingInputRef.current = input }, [input])

  const sendText = React.useCallback(() => {
    const ws = wsRef.current
    if (!ws || ws.readyState !== 1) return
    const text = pendingInputRef.current.trim()
    if (!text) return
    setInput('')
    setAwaitingUserEcho(true)
    ws.send(JSON.stringify({ type:'text_query', text, local_time:new Date().toLocaleString(), muteAudio: muted }))
  }, [muted])

  // ------ rotate session ------
  async function onRotate() {
    try {
      sched.stop()
      const thread = await rotateSession()
      try { wsRef.current?.close(1001, 'rotate') } catch {}
      setTimeout(()=>location.reload(), 100)
      onThreadRotated(thread)
    } catch { alert('Rotate failed') }
  }

  // ------ STOP audio (client -> server) ------
  const onStopAudio = React.useCallback(() => {
    sched.stop()
    const ws = wsRef.current
    if (ws && ws.readyState === 1) {
      ws.send(JSON.stringify({ type: 'stop_audio' }))
    }
  }, [sched])

  // ------ mic / PTT ------
  const {
    micSupported,
    micState,
    requestMicPermission,
    pttDown,
    pttUp,
    pttCancel,
  } = usePTT({
    muted,
    onSend: ({ base64Audio, format }) => {
      const ws = wsRef.current
      if (!ws || ws.readyState !== 1) return
      setAwaitingUserEcho(true)
      ws.send(JSON.stringify({
        type: 'audio_query',
        audio: base64Audio,
        format,
        muteAudio: muted
      }))
    }
  })

  // ------ keyboard ------
  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); if (input.trim()) sendText() }
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'enter') { e.preventDefault(); if (input.trim()) sendText() }
  }

  // ------ toggle mute (notify server immediately) ------
  const toggleMute = React.useCallback(() => {
    const next = !muted
    setMuted(next)
    sched.setMuted(next)
    const ws = wsRef.current
    if (ws && ws.readyState === 1) {
      ws.send(JSON.stringify({ type: next ? 'mute_audio' : 'unmute_audio' }))
    }
  }, [muted, sched])

  // ------ WebSocket wiring ------
  React.useEffect(() => {
    let unmounted = false
    openWS({
      onSocket: (live) => {
        if (wsRef.current && wsRef.current !== live) {
          try { wsRef.current.close(1001, 'superseded') } catch {}
        }
        wsRef.current = live
      },
      onOpen(){ if (!unmounted) setConnected(true) },
      onClose(){ if (!unmounted) { setConnected(false); sched.stop() } },
      onMsg(msg: WSMessage) {
        switch (msg.type) {
          case 'response_start': {
            pendingRef.current = ''
            setPendingAssistant('')
            setStreaming(true)
            setGotFirstToken(false)
            setLiveEmotion(null)
            runEmotionRef.current = null
            break
          }
          case 'emotion': {
            const e = coerceEmotion((msg as any).emotion)
            if (e) {
              runEmotionRef.current = e
              setLiveEmotion(e)
              setLastEmotion(e)
            }
            break
          }
          case 'audio_muted': {
            const m = (msg as any).muted === true
            setMuted(m)
            sched.setMuted(m)
            break
          }
          case 'text_token': {
            const t = (msg as any).token || ''
            if (!t) break
            setPendingAssistant(prev => {
              const next = prev + t
              pendingRef.current = next
              return next
            })
            if (!gotFirstToken) setGotFirstToken(true)
            break
          }
          case 'response_done': {
            const finalText = pendingRef.current.trim()
            if (finalText) {
              const finalEmotion = runEmotionRef.current ? { ...runEmotionRef.current } : undefined
              setMsgs(m => [...m, {
                id: uuid(),
                role:'assistant',
                text: finalText,
                emotion: finalEmotion,
                ts: new Date().toISOString()
              }])
            }
            pendingRef.current = ''
            setPendingAssistant('')
            setStreaming(false)
            setLiveEmotion(null)
            break
          }
          case 'text_query': {
            const txt = (msg as any).text?.toString() ?? ''
            if (txt) setMsgs(m => [...m, { id: uuid(), role:'user', text: txt, ts: new Date().toISOString() }])
            setAwaitingUserEcho(false)
            break
          }
          case 'slides_response': {
            if (!slidesStreamingRef.current) {
              slidesStreamingRef.current = true
              onSlidesStreaming?.(true)
            }
            onSlides((msg as any).slides)
            break
          }
          case 'slides_done': {
            slidesStreamingRef.current = false
            onSlidesStreaming?.(false)
            onSlidesDone?.()
            break
          }
          case 'stop_audio': {
            sched.stop()
            break
          }
          case 'audio_response': {
            const b64 = (msg as any).audio || undefined
            const frames = (msg as any).viseme || []
            const times  = (msg as any).viseme_times || []
            const duration_ms = (msg as any).duration_ms
            // Optional extras the server may include:
            const audio_format = (msg as any).audio_format || 'mp3'
            const chunk_index  = (msg as any).chunk_index
            const offset_ms    = (msg as any).offset_ms
            sched.pushChunk({
              audio: b64,
              viseme: frames,
              viseme_times: times,
              duration_ms,
              audio_format,
              chunk_index,
              offset_ms,
            })
            break
          }
          case 'error': {
            console.error('WS error:', (msg as any).message)
            break
          }
        }
      }
    })
    return () => {
      unmounted = true
      try { wsRef.current?.close(1001, 'unmount') } catch {}
      wsRef.current = null
      sched.stop()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ---------- UI ----------
  const headerEmo = lastEmotion ? (EMOJI[lastEmotion.name] || neutralEmo) : null

  return (
    <div className="flex h-full min-h-0 flex-col gap-3">
      {/* Top bar */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-xs">
          <span className={`inline-flex items-center gap-2 rounded-full px-2 py-1 ${connected ? 'bg-emerald-100 text-emerald-800' : 'bg-slate-200 text-slate-700'}`}>
            <span className={`h-2 w-2 rounded-full ${connected ? 'bg-emerald-500' : 'bg-slate-500'}`} />
            {connected ? 'connected' : 'offline'}
          </span>

          {/* Persisted emotion badge */}
          <span className={`inline-flex items-center gap-1 rounded-full px-2 py-1 ${headerEmo?.pill || neutralEmo.pill} ${headerEmo?.text || neutralEmo.text}`}>
            <span className={`h-2 w-2 rounded-full ${headerEmo?.dot || neutralEmo.dot}`} />
            {(headerEmo?.emoji || neutralEmo.emoji)} {lastEmotion?.name || '—'}
            {lastEmotion?.intensity ? ` ×${Math.max(1,Math.min(3,Math.round(lastEmotion.intensity)))}` : ''}
          </span>

          {micState === 'recording' && (
            <span className="inline-flex items-center gap-2 rounded-full bg-red-100 text-red-700 px-2 py-1">
              <span className="h-2 w-2 rounded-full bg-red-500 animate-pulse" /> recording…
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          {/* mute toggle (indigo) */}
          <button
            onClick={toggleMute}
            className="inline-flex items-center justify-center rounded-full h-10 w-10 bg-indigo-600 text-white hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-300"
            title={muted ? 'Unmute TTS' : 'Mute TTS'}
            aria-label={muted ? 'Unmute' : 'Mute'}
          >
            {muted ? <AiOutlineAudioMuted className="text-2xl" /> : <AiOutlineAudio className="text-2xl" />}
          </button>

          {/* STOP audio (rose) */}
          <button
            onClick={onStopAudio}
            className="inline-flex items-center justify-center rounded-full h-10 w-10 bg-rose-600 text-white hover:bg-rose-700 focus:outline-none focus:ring-2 focus:ring-rose-300"
            title="Stop audio"
            aria-label="Stop audio"
          >
            <AiOutlineStop className="text-2xl" />
          </button>

          {/* rotate (amber) */}
          <button
            onClick={onRotate}
            className="inline-flex items-center justify-center rounded-full h-10 w-10 bg-amber-500 text-white hover:bg-amber-600 focus:outline-none focus:ring-2 focus:ring-amber-300"
            title="Rotate session"
            aria-label="Rotate session"
          >
            <AiOutlineRedo className="text-2xl"/>
          </button>
        </div>
      </div>

      {/* chat history */}
      <div
        ref={scrollRef}
        onScroll={onScroll}
        className="flex-1 min-h-0 overflow-y-auto rounded-2xl border bg-gradient-to-b from-white to-slate-50 p-3 space-y-3"
      >
        {msgs.map((m) => {
          const canon = canonicalEmotionName(m.emotion?.name)
          const style = canon ? EMOJI[canon] : neutralEmo
          const label = canon ?? '—'
          const inten = m.emotion?.intensity ? ` ×${Math.max(1,Math.min(3,Math.round(m.emotion.intensity)))}` : ''
          const ts = m.ts ? new Date(m.ts).toLocaleTimeString() : ''
          const isUser = m.role === 'user'
          return (
            <div key={m.id} className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
              <div className={[
                'group max-w-[80%] rounded-2xl px-3 py-2 shadow-sm border',
                isUser
                  ? 'bg-gradient-to-br from-blue-600 to-blue-500 text-white border-blue-600'
                  : 'bg-white text-slate-800 border-slate-200'
              ].join(' ')}>
                {/* meta */}
                <div className="mb-1 flex items-center justify-between gap-3">
                  {!isUser ? (
                    <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] leading-none ${style.pill} ${style.text}`}>
                      <span className={`h-2 w-2 rounded-full ${style.dot}`} />
                      {style.emoji} {label}{inten}
                    </span>
                  ) : <span className="text-[10px] opacity-70">you</span>}
                  {!!ts && <span className="text-[10px] text-slate-400">{ts}</span>}
                </div>

                {/* content */}
                <div className="prose prose-sm max-w-none prose-a:underline prose-pre:overflow-x-auto">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm, remarkBreaks]}
                    components={{
                      a: (props) => <a {...props} target="_blank" rel="noreferrer" className={isUser ? 'text-white underline underline-offset-2' : 'text-blue-600 underline'} />,
                      code: ({inline, ...props}: any) =>
                        inline ? <code {...props} /> : <pre className="overflow-x-auto"><code {...props} /></pre>,
                    }}
                  >
                    {m.text}
                  </ReactMarkdown>
                </div>
              </div>
            </div>
          )
        })}

        {/* awaiting server echo bubble */}
        {awaitingUserEcho && (
          <div className="flex justify-end">
            <div className="max-w-[70%] rounded-2xl px-3 py-2 bg-blue-500/90 text-white shadow-sm">
              <span className="inline-flex gap-1 items-center">
                <span className="w-1.5 h-1.5 rounded-full bg-white animate-pulse" />
                <span className="w-1.5 h-1.5 rounded-full bg-white animate-pulse" />
                <span className="w-1.5 h-1.5 rounded-full bg-white animate-pulse" />
              </span>
            </div>
          </div>
        )}

        {/* assistant typing/streaming bubble */}
        {streaming && (
          <div className="flex justify-start">
            <div className="max-w-[80%] rounded-2xl px-3 py-2 bg-white text-slate-800 border border-slate-200 shadow-sm">
              {!!liveEmotion && (
                <div className="mb-1">
                  <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] leading-none ${(EMOJI[liveEmotion.name]?.pill || neutralEmo.pill)} ${(EMOJI[liveEmotion.name]?.text || neutralEmo.text)}`}>
                    <span className={`h-2 w-2 rounded-full ${(EMOJI[liveEmotion.name]?.dot || neutralEmo.dot)}`} />
                    {(EMOJI[liveEmotion.name]?.emoji) || neutralEmo.emoji} {liveEmotion.name}
                    {liveEmotion.intensity ? ` ×${Math.max(1,Math.min(3,Math.round(liveEmotion.intensity)))}` : ''}
                  </span>
                </div>
              )}
              <div className="prose prose-sm max-w-none">
                {gotFirstToken
                  ? <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]}>{pendingAssistant}</ReactMarkdown>
                  : <div className="flex items-center gap-2 text-slate-500">
                      <span className="inline-flex gap-1">
                        <span className="w-1.5 h-1.5 rounded-full bg-slate-400 animate-pulse" />
                        <span className="w-1.5 h-1.5 rounded-full bg-slate-400 animate-pulse" />
                        <span className="w-1.5 h-1.5 rounded-full bg-slate-400 animate-pulse" />
                      </span>
                      typing…
                    </div>}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* scroll-to-bottom button */}
      {!atBottom && (
        <div className="flex justify-center -mt-2">
          <button
            onClick={() => { const el = scrollRef.current; if (el) el.scrollTo({ top: el.scrollHeight, behavior:'smooth' }) }}
            className="rounded-full bg-blue-600 text-white text-xs px-3 py-1 shadow hover:bg-blue-700"
          >
            Jump to latest
          </button>
        </div>
      )}

      {/* input row */}
      <div className="sticky bottom-0 flex items-center gap-2">
        <input
          value={input}
          onChange={e=>setInput(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Type a message…"
          className="flex-1 rounded-xl border border-slate-300 p-3 bg-slate-50 text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-400"
          aria-label="Message input"
        />

        {/* send */}
        <button
          onClick={sendText}
          disabled={!connected || !input.trim()}
          className="inline-flex items-center justify-center rounded-xl h-11 w-11 bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 shadow focus:outline-none focus:ring-2 focus:ring-blue-300"
          title={connected ? 'Send' : 'Not connected'}
          aria-label="Send message"
        >
          <AiOutlineSend className="text-2xl" />
        </button>

        {/* mic (PTT) */}
        {/** If micSupported is false, show enable button; else show hold-to-talk */}
        {micSupported !== undefined ? (
          micSupported ? (
            <button
              disabled={!connected}
              onMouseDown={pttDown}
              onMouseUp={pttUp}
              onMouseLeave={pttCancel}
              onTouchStart={pttDown}
              onTouchEnd={pttUp}
              onTouchCancel={pttCancel}
              title="Hold to talk"
              aria-label="Hold to talk"
              className={[
                'inline-flex items-center justify-center rounded-xl h-11 w-11 text-white shadow focus:outline-none focus:ring-2',
                (micState==='recording') ? 'bg-emerald-600 ring-4 ring-emerald-300' : 'bg-emerald-600 hover:bg-emerald-700 focus:ring-emerald-300',
                (!connected) ? 'opacity-50 cursor-not-allowed' : ''
              ].join(' ')}
            >
              <FiMic className="text-2xl"/>
            </button>
          ) : (
            <button
              onClick={requestMicPermission}
              className="inline-flex items-center justify-center rounded-xl h-11 w-11 bg-amber-500 text-white hover:bg-amber-600 shadow focus:outline-none focus:ring-2 focus:ring-amber-300"
              title="Enable microphone"
              aria-label="Enable microphone"
            >
              <AiOutlineUnlock className="text-2xl"/>
            </button>
          )
        ) : null}
      </div>
    </div>
  )
}
