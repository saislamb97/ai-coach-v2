# security/signals.py
import logging
from django.dispatch import receiver
from django.contrib.auth.signals import user_logged_in, user_login_failed
from security.middleware.security import unauthorized_access
from security.utils import get_client_ip, get_ip_location, get_user_agent_info, get_reverse_dns

auth_logger = logging.getLogger("django.security.Authentication")
unauthorized_logger = logging.getLogger("unauthorized.access")
blacklist_logger = logging.getLogger("blacklist")

@receiver(unauthorized_access)
def log_unauthorized_access(sender, request, reason, **kwargs):
    ip = get_client_ip(request)
    loc = get_ip_location(ip)
    ua = get_user_agent_info(request.META.get("HTTP_USER_AGENT", ""))
    rdns = get_reverse_dns(ip)
    msg = (
        f"Unauthorized access: IP={ip}, rDNS={rdns}, "
        f"Loc={loc['city']},{loc['region']},{loc['country']}, ISP={loc['isp']}, "
        f"OS={ua['os']}, App={ua['application']}, Browser={ua['browser']}, Device={ua['device']}, "
        f"URL={request.get_full_path()}, Reason={reason}"
    )
    unauthorized_logger.warning(msg)

@receiver(user_login_failed)
def log_login_failed(sender, credentials, request, **kwargs):
    ip = get_client_ip(request)
    loc = get_ip_location(ip)
    ua = get_user_agent_info(request.META.get("HTTP_USER_AGENT", ""))
    rdns = get_reverse_dns(ip)
    email = (credentials or {}).get("email", "None")
    msg = (
        f"Login failed: IP={ip}, rDNS={rdns}, "
        f"Loc={loc['city']},{loc['region']},{loc['country']}, ISP={loc['isp']}, "
        f"OS={ua['os']}, App={ua['application']}, Browser={ua['browser']}, Device={ua['device']}, "
        f"Email={email}"
    )
    auth_logger.warning(msg)

@receiver(user_logged_in)
def log_login_success(sender, request, user, **kwargs):
    ip = get_client_ip(request)
    loc = get_ip_location(ip)
    ua = get_user_agent_info(request.META.get("HTTP_USER_AGENT", ""))
    rdns = get_reverse_dns(ip)
    msg = (
        f"Login success: IP={ip}, rDNS={rdns}, "
        f"Loc={loc['city']},{loc['region']},{loc['country']}, ISP={loc['isp']}, "
        f"OS={ua['os']}, App={ua['application']}, Browser={ua['browser']}, Device={ua['device']}, "
        f"Email={user.email}"
    )
    auth_logger.info(msg)
