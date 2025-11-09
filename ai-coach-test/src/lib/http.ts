import { ENV } from "./env";

const apiPrefix = (ENV.PROXY_API || "/api").replace(/\/+$/, "");
const base = (ENV.API_BASE || "").replace(/\/+$/, "");

// Build a correct URL whether you pass "/api/..." or just "/agents/..."
function buildUrl(path: string): string {
  const p = path.startsWith("/") ? path : `/${path}`;
  if (base) return `${base}${p}`;
  // No absolute API_BASE → use proxy. If caller already prefixed with /api, keep it.
  return p.startsWith(apiPrefix) ? p : `${apiPrefix}${p}`;
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  if (res.status === 204) return undefined as T;
  const text = await res.text();
  return (text ? JSON.parse(text) : undefined) as T;
}

const commonInit: RequestInit = {
  credentials: "include",
  headers: {
    Accept: "application/json",
    Authorization: `Api-Key ${ENV.API_KEY}`,
  },
};

export async function get<T = any>(path: string): Promise<T> {
  const res = await fetch(buildUrl(path), { ...commonInit, method: "GET" });
  return handle<T>(res);
}

export async function post<T = any>(path: string, body?: any): Promise<T> {
  const res = await fetch(buildUrl(path), {
    ...commonInit,
    method: "POST",
    headers: { ...(commonInit.headers as any), "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  return handle<T>(res);
}

export async function patch<T = any>(path: string, body?: any): Promise<T> {
  const res = await fetch(buildUrl(path), {
    ...commonInit,
    method: "PATCH",
    headers: { ...(commonInit.headers as any), "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  // Optional: transparent PUT fallback if the API doesn't allow PATCH
  if (res.status === 405) {
    const res2 = await fetch(buildUrl(path), {
      ...commonInit,
      method: "PUT",
      headers: { ...(commonInit.headers as any), "Content-Type": "application/json" },
      body: JSON.stringify(body ?? {}),
    });
    return handle<T>(res2);
  }
  return handle<T>(res);
}

export async function put<T = any>(path: string, body?: any): Promise<T> {
  const res = await fetch(buildUrl(path), {
    ...commonInit,
    method: "PUT",
    headers: { ...(commonInit.headers as any), "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  return handle<T>(res);
}

export async function del<T = any>(path: string): Promise<T> {
  const res = await fetch(buildUrl(path), { ...commonInit, method: "DELETE" });
  return handle<T>(res);
}
