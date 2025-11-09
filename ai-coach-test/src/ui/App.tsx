// components/App.tsx
import React from 'react'
import { ENV } from '../lib/env'
import { ensureSession } from '../lib/session'
import { fetchAgent, saveAgentGlb } from '../lib/agent'
import AvatarCanvas from './AvatarCanvas'
import SlidesPane from './SlidesPane'
import ChatPane from './ChatPane'
import RPMEmbed from './RPMEmbed'

const LS_GLB = 'glb_url'

function Chip({ children }: { children: React.ReactNode }) {
  return (
    <span className="inline-flex items-center rounded-full bg-slate-100 px-2 py-0.5 text-[11px] font-medium text-slate-600 ring-1 ring-slate-200">
      {children}
    </span>
  )
}
function Skeleton({ className='' }: { className?: string }) {
  return <div className={`animate-pulse rounded bg-slate-200/70 ${className}`} />
}

function isTextEditingTarget(t: EventTarget | null) {
  const el = t as HTMLElement | null
  if (!el) return false
  const tag = el.tagName?.toLowerCase()
  // @ts-ignore
  const editable = !!el.isContentEditable
  return tag === 'input' || tag === 'textarea' || editable
}

export default function App() {
  // --- boot / state ---
  const [threadId, setThreadId] = React.useState<string>('')
  const [sessionReady, setSessionReady] = React.useState(false)

  const [glbUrl, setGlbUrl] = React.useState<string>('') // drives AvatarCanvas
  const [rpmOpen, setRpmOpen] = React.useState(false)

  const [bootError, setBootError] = React.useState<string | null>(null)
  const [loadingAgent, setLoadingAgent] = React.useState(true)

  // slides streaming state from WS
  const [wsSlides, setWsSlides] = React.useState<any|null>(null)
  const [slidesStreaming, setSlidesStreaming] = React.useState(false)

  // visemes wiring for AvatarCanvas
  const lastFrame = React.useRef<number[] | null>(null)
  const getVisemeFrameRef = React.useRef<() => number[] | null>(() => lastFrame.current)
  const getVisemeFrame = React.useCallback(
    () => (getVisemeFrameRef.current ? getVisemeFrameRef.current() : null),
    []
  )

  // --- ensure session first ---
  React.useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        setBootError(null)
        const t = await ensureSession()
        if (cancelled) return
        setThreadId(t)
        setSessionReady(true)
      } catch (e: any) {
        setBootError(e?.message || 'Failed to initialize session')
      }
    })()
    return () => { cancelled = true }
  }, [])

  // --- load agent avatar (server → LS fallback) ---
  React.useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        setLoadingAgent(true)
        const fromLs = localStorage.getItem(LS_GLB) || ''
        const agent = await fetchAgent().catch(() => null)
        if (cancelled) return
        const serverGlb = agent?.glb_url || ''
        const chosen = serverGlb || fromLs
        if (chosen) setGlbUrl(chosen)
        else setRpmOpen(true)
      } catch {
        const fromLs = localStorage.getItem(LS_GLB) || ''
        if (fromLs) setGlbUrl(fromLs)
        else setRpmOpen(true)
      } finally {
        if (!cancelled) setLoadingAgent(false)
      }
    })()
    return () => { cancelled = true }
  }, [])

  // --- SAFE keyboard shortcut: Ctrl/Cmd + Shift + A (ignore while typing) ---
  React.useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const k = e.key?.toLowerCase()
      const modOk = (e.ctrlKey || e.metaKey) && e.shiftKey
      if (k === 'a' && modOk && !isTextEditingTarget(e.target)) {
        e.preventDefault()
        setRpmOpen(true)
      }
    }
    window.addEventListener('keydown', onKey, { passive: false })
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  // --- RPM export handler ---
  async function onAvatarExport(glb: string) {
    try {
      setGlbUrl(glb)
      localStorage.setItem(LS_GLB, glb)
      await saveAgentGlb(glb).catch(() => {})
    } finally {
      setRpmOpen(false)
    }
  }

  return (
    // FULL SCREEN, compact bar only
    <div className="h-dvh w-dvw bg-slate-50">
      <div className="flex h-full flex-col">
        {/* Compact top bar: IDs/labels only */}
        <div className="shrink-0 border-b border-slate-200 bg-white">
          <div className="mx-auto max-w-7xl px-3 md:px-4 py-2 flex items-center justify-between gap-2">
            <div className="flex flex-wrap items-center gap-2">
              <Chip>bot: <span className="ml-1 font-mono">{ENV.BOT_ID || '—'}</span></Chip>
              <Chip>lang: <span className="ml-1 font-mono">{ENV.LANG}</span></Chip>
              <Chip>thread: <span className="ml-1 font-mono">{threadId || '…'}</span></Chip>
            </div>
            <div className="flex items-center gap-2">
              {!!bootError && (
                <button
                  onClick={() => location.reload()}
                  className="text-xs rounded px-2 py-1 border border-rose-200 text-rose-700 bg-rose-50 hover:bg-rose-100"
                  title="Retry boot"
                >
                  Retry
                </button>
              )}
              <button
                onClick={() => setRpmOpen(true)}
                className="rounded-lg bg-emerald-600 text-white px-3 py-1.5 text-xs hover:bg-emerald-700"
                title="Customize Avatar (Ctrl/Cmd + Shift + A)"
              >
                Customize Avatar
              </button>
            </div>
          </div>
        </div>

        {/* BODY fills remaining space */}
        <main className="flex-1 min-h-0 p-3 md:p-4">
          <div className="mx-auto h-full max-w-7xl grid min-h-0 grid-rows-1 md:grid-cols-12 gap-3 md:gap-4">
            {/* LEFT column: Avatar (1) + Chat (2) → chat taller */}
            <section className="min-h-0 md:col-span-5 grid grid-rows-[1fr_2fr] gap-3">
              {/* Avatar */}
              <div className="rounded-2xl bg-white ring-1 ring-black/5 p-3 min-h-0">
                {loadingAgent ? (
                  <>
                    <Skeleton className="h-full w-full" />
                    <div className="mt-3 text-xs text-slate-500">Loading avatar…</div>
                  </>
                ) : (
                  <div className="h-full min-h-[200px]">
                    <AvatarCanvas glbUrl={glbUrl} getVisemeFrame={getVisemeFrame} className="h-full" zoom={0.3} />
                  </div>
                )}
              </div>

              {/* Chat */}
              <div className="rounded-2xl bg-white ring-1 ring-black/5 p-3 min-h-0">
                {sessionReady ? (
                  <div className="h-full min-h-0">
                    <ChatPane
                      onSlides={(s) => setWsSlides(s)}
                      onSlidesStreaming={(on) => setSlidesStreaming(on)}
                      onSlidesDone={() => setSlidesStreaming(false)}
                      getVisemeFrameRef={getVisemeFrameRef}
                      onThreadRotated={(t) => setThreadId(t)}
                    />
                  </div>
                ) : (
                  <div>
                    <Skeleton className="h-6 w-40" />
                    <Skeleton className="mt-3 h-64 w-full" />
                    <div className="mt-2 text-xs text-slate-500">Preparing session…</div>
                  </div>
                )}
              </div>
            </section>

            {/* RIGHT column: Slides */}
            <section className="min-h-0 md:col-span-7 rounded-2xl bg-white ring-1 ring-black/5 p-3 md:p-4 flex flex-col">
              <div className="mb-2 flex items-center justify-between">
                <div className="text-sm font-semibold text-slate-700">Main Slide</div>
                <div className="text-xs text-slate-400">Autosaves while you type</div>
              </div>
              <div className="flex-1 min-h-0">
                <SlidesPane className="h-full"
                  incoming={wsSlides}
                  streaming={slidesStreaming}
                />
              </div>
            </section>
          </div>
        </main>

        {/* RPM Modal */}
        {rpmOpen && (
          <div className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-4">
            <div className="bg-white rounded-2xl w-full max-w-5xl p-4 space-y-3 shadow-2xl">
              <div className="flex items-center justify-between">
                <div className="font-semibold">Ready Player Me</div>
                <button
                  onClick={() => setRpmOpen(false)}
                  className="text-sm rounded px-3 py-1 border hover:bg-slate-700 bg-slate-500"
                >
                  Close
                </button>
              </div>
              <RPMEmbed
                src={ENV.RPM_FRAME_URL}
                onReady={() => {}}
                onExported={onAvatarExport}
                className="w-full h-[72vh] rounded-xl border"
              />
              <div className="text-xs text-slate-500">
                Build your avatar. When you click <b>Export</b>, it loads automatically and is saved (server + local).
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
