export const ENV = {
  API_BASE: (import.meta.env.VITE_API_BASE || "").replace(/\/+$/, ""),
  WS_BASE:  (import.meta.env.VITE_WS_BASE  || "").replace(/\/+$/, ""), // optional direct ws base; usually leave empty to use proxy
  API_KEY:  import.meta.env.VITE_API_KEY || "",
  BOT_ID:   import.meta.env.VITE_BOT_ID  || "",
  LANG:     import.meta.env.VITE_WEBSITE_LANGUAGE || "en",
  RPM_FRAME_URL: (import.meta.env.VITE_RPM_FRAME_URL as string) || "https://readyplayer.me/avatar?frameApi",
  PROXY_API: import.meta.env.VITE_PROXY_API || "/api",
  PROXY_WS:  import.meta.env.VITE_PROXY_WS  || "/ws",
};

// Warn if critical bits are missing
if (!ENV.BOT_ID)  console.warn("VITE_BOT_ID missing");
if (!ENV.API_KEY) console.warn("VITE_API_KEY missing");

// REST auth via cookie (your DRF views accept it)
try {
  const secure = location.protocol === "https:" ? "; Secure" : "";
  document.cookie = `api_key=${encodeURIComponent(ENV.API_KEY)}; path=/; SameSite=Lax${secure}`;
} catch {}
