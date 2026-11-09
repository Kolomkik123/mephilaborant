"""Select the persistent team index or the original calculator index."""
import os

def make_retriever(backend=None):
    chosen=backend or os.getenv('RETRIEVAL_BACKEND','team')
    if chosen=='team':
        from app.team_search.adapter import TeamRetriever
        return TeamRetriever()
    if chosen=='legacy':
        from .retriever import Retriever
        return Retriever()
    raise ValueError('RETRIEVAL_BACKEND должен быть team или legacy')
