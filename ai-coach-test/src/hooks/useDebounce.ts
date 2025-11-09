import { useEffect, useRef } from 'react'

export function useDebounce(fn: () => void, deps: any[], ms = 800) {
  const t = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (t.current) {
      clearTimeout(t.current)
    }

    t.current = window.setTimeout(fn, ms)

    return () => {
      if (t.current) {
        clearTimeout(t.current)
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
}
