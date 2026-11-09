"""Convert colleague's Source records into the calculator's Retrieval/Hit."""
from pathlib import Path
from app.data.parse import Fragment
from app.data.index import Hit
from app.data.retriever import Retrieval
from app.data.index import content_terms
from .search import search_manuals, fragments_by_work

def adapt(query, sources):
    hits=[Hit(fragment=Fragment(text=item.text,source=Path(item.source_file).stem,
         page=item.page,work=item.work),score=0,dense_score=item.similarity) for item in sources]
    return Retrieval(query=query,hits=hits,relevant=bool(hits),
                     dense_score=max((hit.dense_score for hit in hits),default=0))

class TeamRetriever:
    def fragments_for(self,work_or_topic,limit=5,work="",kind="text"):
        if kind!="text":
            raise ValueError("Поисковик команды пока предоставляет текстовые фрагменты, не отдельный банк вопросов")
        if work:
            # Formula selection must see the whole selected lab, including later pages.
            return adapt(work_or_topic,fragments_by_work(work,None))
        return self.search(work_or_topic,limit,kind)

    def search(self,query,top_k=5,kind="text"):
        if kind not in ("text",None):
            raise ValueError("Поддержан поиск текстовых фрагментов")
        sources=search_manuals(query,k=max(15,top_k*3))
        terms=content_terms(query)
        # Keep the colleague's neural candidates, then prefer literal matches
        # to Russian query terms. Synonym-only results retain their vector order.
        if terms:
            sources.sort(key=lambda item:(len(terms & content_terms(item.text))/len(terms),item.similarity),reverse=True)
        return adapt(query,sources[:top_k])
