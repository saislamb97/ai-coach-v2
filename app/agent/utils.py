from django.utils.text import slugify

# ===========================
#         Path Utils
# ===========================
def agent_upload_path(instance, filename):
    safe_name = slugify(getattr(instance, "name", "") or "agent")
    return f"agent/{safe_name}/{filename}"

def voice_upload_path(instance, filename):
    safe_name = slugify(getattr(instance, "name", "") or "voice")
    return f"voice/{safe_name}/{filename}"