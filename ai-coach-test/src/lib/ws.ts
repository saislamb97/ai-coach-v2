import { ENV } from "./env";
import { getThreadId, ensureSession } from "./session";

export type WSMessage =
  | { type: "connected"; bot_id: string; thread_id: string }
  | { type: "text_query"; text: string }
  | { type: "response_start" }
  | { type: "emotion"; emotion: { name: string; intensity: number } }
  | { type: "text_token"; token: string }
  | {
      type: "audio_response";
      audio: string;
      viseme: number[][];
      viseme_times: number[];
      viseme_format?: string;
      frame_ms?: number;
      duration_ms?: number;
    }
  | { type: "slides_response"; slides: any }
  | { type: "slides_done" }
  | { type: "response_done"; timings?: any }
  | { type: "pong" }
  | { type: "error"; message: string }
  | { type: string; [k: string]: any };

export type WSHandlers = {
  onSocket?: (ws: WebSocket) => void;
  onOpen?: () => void;
  onClose?: (code?: number) => void;
  onMsg?: (msg: WSMessage) => void;
};

function sameOriginBase(): string {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${location.host}`;
}
function directBackendBase(): string {
  if (ENV.WS_BASE && ENV.WS_BASE.length) return ENV.WS_BASE.replace(/\/+$/, "");
  const api = (ENV.API_BASE || "").replace(/\/+$/, "");
  if (!api) return "";
  return api.replace(/^http:/, "ws:").replace(/^https:/, "wss:");
}

async function ensureThread(): Promise<string> {
  let tid = getThreadId();
  if (tid) return tid;
  try { tid = (await ensureSession()) || ""; } catch {}
  return getThreadId() || tid || "";
}

function qs(thread_id: string): string {
  const q = new URLSearchParams({
    bot_id: ENV.BOT_ID,
    thread_id,
    website_language: ENV.LANG || "en",
  });
  if (ENV.API_KEY) q.set("api_key", ENV.API_KEY);
  return q.toString();
}

function proxyUrl(thread_id: string): string {
  const base = sameOriginBase();
  const root = ENV.PROXY_WS || "/ws";
  return `${base}${root}/chat/?${qs(thread_id)}`;
}
function directUrl(thread_id: string): string {
  const base = directBackendBase();
  if (!base) return "";
  return `${base}/ws/chat/?${qs(thread_id)}`;
}

const HANDSHAKE_FAILURES = new Set([1006, 4403, 4000, 4004, 4005]);

export function openWS(h: WSHandlers) {
  let closed = false;
  let active: WebSocket | null = null;

  (async () => {
    const tid = await ensureThread();
    const candidates = [proxyUrl(tid), directUrl(tid)].filter(Boolean) as string[];
    let idx = 0;

    function tryNext(prev?: { code?: number; reason?: string }) {
      if (closed) return;
      const url = candidates[idx++];
      if (!url) {
        console.error("[WS] all endpoints failed", prev);
        h.onClose?.(prev?.code);
        return;
      }
      const ws = new WebSocket(url);
      active = ws;
      h.onSocket?.(ws);

      ws.onopen = () => h.onOpen?.();
      ws.onmessage = (ev) => {
        try { h.onMsg?.(JSON.parse(ev.data)); } catch {}
      };
      ws.onerror = (e) => console.warn("[WS] error", url, e);
      ws.onclose = (ev) => {
        if (!closed && HANDSHAKE_FAILURES.has(ev.code) && idx < candidates.length) {
          tryNext({ code: ev.code, reason: (ev as any).reason });
        } else {
          h.onClose?.(ev.code);
        }
      };
    }

    tryNext();
  })();

  if (!active) {
    console.warn("⚠️ WebSocket not connected yet");
    return null;
  }
  return active;
}
