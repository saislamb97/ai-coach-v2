import { useEffect, useRef } from 'react'

export function useDebounce(fn: () => void, deps: any[], ms=800) {
  const t = useRef<number>()
  useEffect(() => {
    window.clearTimeout(t.current)
    t.current = window.setTimeout(() => fn(), ms)
    return () => window.clearTimeout(t.current)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
}
