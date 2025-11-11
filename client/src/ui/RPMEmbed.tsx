import React from 'react'

type Props = {
  src: string
  onExported: (glbUrl: string) => void
  onReady?: () => void
  className?: string
}

export default function RPMEmbed({ src, onExported, onReady, className }: Props) {
  const frameRef = React.useRef<HTMLIFrameElement|null>(null)

  React.useEffect(() => {
    function onMessage(event: MessageEvent) {
      let data: any
      try { data = typeof event.data === 'string' ? JSON.parse(event.data) : event.data } catch { return }
      if (!data || data.source !== 'readyplayerme') return

      if (data.eventName === 'v1.frame.ready') {
        frameRef.current?.contentWindow?.postMessage(
          JSON.stringify({ target: 'readyplayerme', type: 'subscribe', eventName: 'v1.**' }), '*'
        )
        onReady?.()
      }
      if (data.eventName === 'v1.avatar.exported' && data.data?.url) {
        onExported(String(data.data.url))
      }
    }
    window.addEventListener('message', onMessage)
    return () => window.removeEventListener('message', onMessage)
  }, [onExported, onReady])

  return (
    <iframe
      ref={frameRef}
      className={className || 'w-full h-[70vh] rounded-xl border border-gray-200'}
      allow="camera *; microphone *; clipboard-write"
      src={src}
      title="Ready Player Me"
    />
  )
}
