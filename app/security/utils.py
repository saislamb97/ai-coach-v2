# security/utils.py
import requests, socket
from user_agents import parse

def get_client_ip(request):
    x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
    return x_forwarded_for.split(",")[0].strip() if x_forwarded_for else request.META.get("REMOTE_ADDR")

def get_ip_location(ip, timeout=3):
    try:
        r = requests.get(f"https://ipinfo.io/{ip}/json", timeout=timeout)
        j = r.json()
        org = j.get("org", "Unknown")
        return {
            "city": j.get("city", "Unknown"),
            "region": j.get("region", "Unknown"),
            "country": j.get("country", "Unknown"),
            "org": org,
            "isp": org,
        }
    except Exception:
        return {"city":"Unknown","region":"Unknown","country":"Unknown","org":"Unknown","isp":"Unknown"}

def get_reverse_dns(ip):
    try:
        return socket.gethostbyaddr(ip)[0]
    except Exception:
        return "Unknown Hostname"

def get_user_agent_info(ua_string):
    ua = parse(ua_string or "")
    return {
        "os": ua.os.family,
        "browser": ua.browser.family,
        "device": ua.device.family,
        "brand": ua.device.brand,
        "model": ua.device.model,
        "application": detect_application(ua_string or ""),
    }

def detect_application(ua_string: str) -> str:
    s = ua_string.lower()
    if "unity" in s: return "Unity Game Engine"
    if "unreal" in s: return "Unreal Engine"
    if "android" in s: return "Android APK"
    if "iphone" in s or "ios" in s: return "iPhone App"
    if "huawei" in s: return "Huawei App"
    if "smarttv" in s or " tv " in s: return "TV OS"
    if "windows" in s: return "Windows App"
    if "macintosh" in s or "mac os" in s: return "Mac OS App"
    if "linux" in s: return "Linux App"
    if "chrome os" in s: return "Chrome OS App"
    if "firefox" in s: return "Firefox Browser"
    if "safari" in s: return "Safari Browser"
    if "edge" in s: return "Edge Browser"
    if "opera" in s: return "Opera Browser"
    if "samsungbrowser" in s: return "Samsung Browser"
    return "Unknown Application"
