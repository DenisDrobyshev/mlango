"""Inference API routes — the ``urls.py`` of an mlango project."""

from mlango.serve import path

from demo.agents import Helper
from demo.models import Sentiment

urlpatterns = [
    # POST /api/predict/  {"input": "great movie"}
    path("predict/", Sentiment.as_endpoint(), name="sentiment-predict"),
    # POST /api/chat/     {"message": "hello", "session_id": "abc"}
    path("chat/", Helper.as_endpoint(), name="helper-chat"),
]
