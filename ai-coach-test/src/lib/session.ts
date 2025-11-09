import { post } from './http'
import { ENV } from './env'

const LS_KEY = 'thread_id';

export function getThreadId(): string | null {
  return localStorage.getItem(LS_KEY);
}

export function setThreadId(t: string) {
  localStorage.setItem(LS_KEY, t);
}

export async function ensureSession(): Promise<string> {
  const existing = getThreadId();
  if (existing) return existing;

  // Create session via API (serializer resolves agent by bot_id)
  const resp = await post('/api/sessions/', { bot_id: ENV.BOT_ID });
  // SessionReadSerializer returns thread_id
  const thread = resp.thread_id as string;
  setThreadId(thread);
  return thread;
}

export async function rotateSession(): Promise<string> {
  localStorage.removeItem(LS_KEY);
  return ensureSession();
}
