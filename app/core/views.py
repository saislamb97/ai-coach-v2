# core/views.py
from __future__ import annotations
from django.shortcuts import render

def apiDoc(request):
    """
    Public api doc page for REST APIs and WebSocket.
    Pulls no dynamic data; safe for DEBUG.
    """
    return render(request, "api_doc.html", {})
