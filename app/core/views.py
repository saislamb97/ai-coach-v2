# core/views.py
from __future__ import annotations
from django.shortcuts import render

def home(request):
    """
    Public home page for Demo.
    """
    return render(request, "index.html", {})
