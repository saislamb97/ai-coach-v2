import React from 'react'
import type { ToolConstructable } from '@editorjs/editorjs'
import EditorJS from '@editorjs/editorjs'
import type { OutputData } from '@editorjs/editorjs'
import Header from '@editorjs/header'
import List from '@editorjs/list'
import Paragraph from '@editorjs/paragraph'
import { useDebounce } from '../hooks/useDebounce'
import { get, post } from '../lib/http'
import { getThreadId } from '../lib/session'

type Slides = {
  id?: number
  thread_id?: string
  version?: number
  title: string
  summary: string
  editorjs: OutputData
  updated_by?: string
  updated_at?: string
}

type Props = {
  className?: string
  /** Live WS chunk (entire slides payload or partial) */
  incoming?: any | null
  /** Show “Streaming…” badge */
  streaming?: boolean
}

/* ---------------- utilities ---------------- */
function stableHash(obj: any): string {
  const norm = JSON.stringify(obj, (_k, v) => {
    if (v && typeof v === 'object' && !Array.isArray(v)) {
      return Object.keys(v).sort().reduce((acc: any, k) => { acc[k] = v[k]; return acc }, {})
    }
    return v
  })
  let h = 5381
  for (let i = 0; i < norm.length; i++) h = ((h << 5) + h) ^ norm.charCodeAt(i)
  return (h >>> 0).toString(36)
}
const nowFmt = () => new Date().toLocaleTimeString()

/** Returns true if the EditorJS data has any meaningful user content */
function editorHasContent(data?: OutputData | null): boolean {
  if (!data || !Array.isArray((data as any).blocks)) return false
  const blocks = (data as any).blocks as Array<any>
  if (!blocks.length) return false

  for (const b of blocks) {
    if (!b || typeof b !== 'object') continue
    const t = String(b.type || '').toLowerCase()
    const d = (b.data ?? {}) as any

    if (t === 'header') {
      const text = String(d.text ?? '').trim()
      if (text) return true
    } else if (t === 'paragraph') {
      const text = String(d.text ?? '').replace(/<[^>]*>/g, '').trim()
      if (text) return true
    } else if (t === 'list') {
      const items = Array.isArray(d.items) ? d.items : []
      if (items.some((x: any) => String(x ?? '').trim().length > 0)) return true
    } else {
      // Any other block type with some data counts as content
      if (Object.values(d).some((v) => (typeof v === 'string' ? v.trim().length > 0 : !!v))) return true
    }
  }
  return false
}

export default function SlidesPane({ className = '', incoming, streaming = false }: Props) {
  const holderId = React.useMemo(
    () => `slides-editor-${(crypto as any)?.randomUUID?.() ?? Math.random().toString(36).slice(2)}`,
    []
  )

  const [title, setTitle] = React.useState('')
  const [summary, setSummary] = React.useState('')
  const [saving, setSaving] = React.useState(false)
  const [applying, setApplying] = React.useState(false)
  const [savedAt, setSavedAt] = React.useState<string | null>(null)
  const silenceAutosaveRef = React.useRef(false)

  // versioning / conflict awareness
  const versionRef = React.useRef<number>(0)
  const lastSavedHashRef = React.useRef<string>('')

  // user activity (avoid clobbering caret while typing)
  const lastEditTsRef = React.useRef<number>(0)
  const markEdited = () => {
    lastEditTsRef.current = Date.now()
  }
  const userActive = () => Date.now() - lastEditTsRef.current < 900 // ms

  // queued AI updates (when user is active)
  const [queuedAI, setQueuedAI] = React.useState<Slides | null>(null)

  const ej = React.useRef<EditorJS | null>(null)
  const [ejData, setEjData] = React.useState<OutputData>({
    time: Date.now(),
    version: '2.x',
    blocks: [],
  })
  const readyRef = React.useRef(false) // becomes true after initial fetch/attempt completes

  // ------------- init -------------
  React.useEffect(() => {
    let mounted = true
    ej.current = new EditorJS({
      holder: holderId,
      autofocus: true,
      data: ejData,
      minHeight: 0,
      tools: {
        header: {
          class: Header as unknown as ToolConstructable,
          inlineToolbar: true,
          config: { levels: [2, 3], defaultLevel: 2 },
        },
        list: { class: List as unknown as ToolConstructable, inlineToolbar: true },
        paragraph: {
          class: Paragraph as unknown as ToolConstructable,
          inlineToolbar: true,
        },
      },
      onChange: async () => {
        if (!mounted || !ej.current) return
        if (silenceAutosaveRef.current) return
        try {
          const data = await ej.current.save()
          setEjData(data)
          markEdited()
        } catch {}
      },
    })

    ;(async () => {
      const thread_id = getThreadId()
      if (!thread_id) {
        // No thread: mark ready so autosave guard can still run, but doSave() will return early due to no thread id.
        readyRef.current = true
        return
      }
      try {
        const qs = new URLSearchParams({ thread_id, ordering: '-updated_at' }).toString()
        const res = await get<any>(`/api/slides/?${qs}`)
        const pickLatest = (r: any): Slides | null => {
          if (!r) return null
          if (Array.isArray(r?.results)) return r.results[0] || null
          if (Array.isArray(r)) return r[0] || null
          if (r?.editorjs) return r as Slides
          return null
        }
        const latest = pickLatest(res)
        if (mounted && latest?.editorjs) {
          versionRef.current = Number(latest.version || 0)
          setTitle(latest.title || '')
          setSummary(latest.summary || '')
          silenceAutosaveRef.current = true
          try {
            await ej.current?.isReady
            await ej.current?.render(latest.editorjs)
            setEjData(latest.editorjs)
          } finally {
            setTimeout(() => {
              silenceAutosaveRef.current = false
            }, 0)
          }
          const hash = stableHash({ t: latest.title || '', s: latest.summary || '', e: latest.editorjs })
          lastSavedHashRef.current = hash
          setSavedAt(nowFmt())
        }
      } finally {
        // Mark ready after fetch attempt completes (whether or not data existed)
        readyRef.current = true
      }
    })()

    return () => {
      mounted = false
      try {
        ej.current?.destroy()
      } catch {}
      ej.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [holderId])

  // ------------- incoming WS updates -------------
  React.useEffect(() => {
    if (!incoming) return
    const next: Slides = {
      title: String(incoming.title ?? title ?? ''),
      summary: String(incoming.summary ?? summary ?? ''),
      editorjs:
        incoming.editorjs && typeof incoming.editorjs === 'object'
          ? (incoming.editorjs as OutputData)
          : ejData,
      version: Number(incoming.version || versionRef.current) || versionRef.current,
    }

    // Ignore stale versions
    if ((next.version ?? 0) <= (versionRef.current ?? 0)) return

    // If user is actively editing, queue the update instead of applying
    if (userActive()) {
      setQueuedAI(next)
      return
    }

    // Apply immediately
    setApplying(true)
    silenceAutosaveRef.current = true
    ;(async () => {
      try {
        versionRef.current = next.version ?? versionRef.current
        setTitle(next.title || '')
        setSummary(next.summary || '')
        await ej.current?.isReady
        await ej.current?.render(next.editorjs)
        setEjData(next.editorjs)
      } finally {
        setApplying(false)
        setTimeout(() => {
          silenceAutosaveRef.current = false
        }, 0)
      }
    })()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [incoming])

  // ------------- apply / discard queued AI -------------
  const applyQueued = React.useCallback(async () => {
    const q = queuedAI
    if (!q) return
    setQueuedAI(null)
    setApplying(true)
    silenceAutosaveRef.current = true
    try {
      versionRef.current = q.version ?? versionRef.current + 1
      setTitle(q.title || '')
      setSummary(q.summary || '')
      await ej.current?.isReady
      await ej.current?.render(q.editorjs)
      setEjData(q.editorjs)
    } finally {
      setApplying(false)
      setTimeout(() => {
        silenceAutosaveRef.current = false
      }, 0)
    }
  }, [queuedAI])

  const discardQueued = React.useCallback(() => {
    setQueuedAI(null)
  }, [])

  // ------------- debounced autosave -------------
  const doSave = React.useCallback(async () => {
    if (!readyRef.current || silenceAutosaveRef.current) return
    const thread_id = getThreadId()
    if (!thread_id) return

    // ⛔️ Skip saving if slide is effectively empty
    const noTitle = !title.trim()
    const noSummary = !summary.trim()
    const noBlocks = !editorHasContent(ejData)
    if (noTitle && noSummary && noBlocks) return

    const payload = {
      thread_id,
      title: title || 'Untitled Deck',
      summary: summary || '',
      editorjs: ejData,
      rotate: false,
      updated_by: 'ui',
      version: versionRef.current,
    }
    const hash = stableHash({ t: payload.title, s: payload.summary, e: payload.editorjs })
    if (hash === lastSavedHashRef.current) return // no-op change

    setSaving(true)
    try {
      await post('/api/slides/', payload)
      lastSavedHashRef.current = hash
      setSavedAt(nowFmt())
    } catch {
      // optional: toast
    } finally {
      setSaving(false)
    }
  }, [title, summary, ejData])

  useDebounce(() => {
    void doSave()
  }, [title, summary, ejData])

  const Sep = () => <span className="text-slate-300">•</span>

  return (
    <div className={`flex h-full min-h-0 flex-col gap-2 ${className}`}>
      {/* Compact single-line header */}
      <div className="flex items-center gap-2 text-xs whitespace-nowrap overflow-hidden min-h-[28px]">
        {streaming && (
          <span className="inline-flex items-center gap-1 rounded-full bg-blue-100 text-blue-800 px-2 py-[2px]">
            <span className="relative inline-flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-600"></span>
            </span>
            Streaming…
          </span>
        )}
        {applying && (
          <>
            <Sep />
            <span className="inline-flex items-center gap-1 rounded-full bg-emerald-100 text-emerald-800 px-2 py-[2px]">
              <span className="h-2 w-2 rounded-full bg-emerald-600 animate-pulse"></span>
              Updating…
            </span>
          </>
        )}
        {!!queuedAI && (
          <>
            <Sep />
            <span className="inline-flex items-center gap-1 rounded-full bg-violet-100 text-violet-800 px-2 py-[2px]">
              <span className="h-2 w-2 rounded-full bg-violet-600 animate-pulse"></span>
              AI update ready
              <button
                onClick={applyQueued}
                className="ml-1 rounded bg-violet-600 text-white px-1.5 py-[1px] hover:bg-violet-700"
              >
                Apply
              </button>
              <button
                onClick={discardQueued}
                className="rounded border border-violet-300 px-1.5 py-[1px] hover:bg-violet-50"
              >
                Discard
              </button>
            </span>
          </>
        )}
        <Sep />
        <span className="inline-flex items-center rounded-full bg-slate-100 px-2 py-[2px] text-[11px] text-slate-600 ring-1 ring-slate-200">
          v{versionRef.current}
        </span>
        <Sep />
        {saving ? (
          <span className="inline-flex items-center gap-1 text-slate-600">
            <span className="h-2 w-2 rounded-full bg-slate-400 animate-pulse"></span>
            Saving…
          </span>
        ) : (
          <span className="text-slate-400 truncate">Saved{savedAt ? ` @ ${savedAt}` : ''}</span>
        )}
      </div>

      {/* Title/summary */}
      <div className="grid grid-cols-1 gap-2">
        <input
          className="rounded border border-slate-300 bg-white text-slate-900 placeholder-slate-400 p-2 text-base font-semibold"
          placeholder="Slide title"
          value={title}
          onChange={(e) => {
            setTitle(e.target.value)
            markEdited()
          }}
        />
        <textarea
          className="rounded border border-slate-300 bg-white text-slate-900 placeholder-slate-400 p-2 text-sm"
          placeholder="Short description"
          rows={3}
          value={summary}
          onChange={(e) => {
            setSummary(e.target.value)
            markEdited()
          }}
        />
      </div>

      {/* Editor area fills & scrolls */}
      <div className="flex-1 min-h-0 rounded border border-slate-200 bg-slate-50">
        <div className="h-full min-h-0 overflow-auto">
          <div id={holderId} className="min-h-[320px] p-3 text-slate-900" />
        </div>
      </div>

      <div className="text-[11px] text-slate-500">
        Autosaves while you type. Live changes from the assistant stream in real time.
      </div>
    </div>
  )
}
