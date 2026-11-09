"""End-to-end checks using the real local MiniLM/Chroma index."""
import json
import math
from datetime import datetime, timezone
from app.config import ROOT
from app.agents.base import AgentContext, DummyTrace
from app.agents.calculator import Calculator
from .adapter import TeamRetriever
from .search import index_manifest,search_manuals,fragments_by_work

def verify():
    records=[]
    retriever=TeamRetriever()
    scenarios=[
        ("atwood",{"question":"найди ускорение","work":"1.2","mode":"fit","x_column":"t^2","y_column":"s",
         "table_text":"S, м; t², с²\n0.20;0.41\n0.40;0.80\n0.60;1.22\n0.80;1.59\n1.00;2.01"},1.00218,"a"),
        ("pendulum",{"question":"L = 1 м","work":"1.3","mode":"repeat",
         "table_text":"T, с\n2\n2.01\n1.99\n2.02\n1.98"},math.pi**2,"g")]
    for name,state,expected,symbol in scenarios:
        result=Calculator(AgentContext(DummyTrace()),retriever=retriever).process(state)
        indirect=result.get("payload",{}).get("indirect",{})
        passed=(not result.get("refused") and indirect.get("quantity")==symbol
                and indirect.get("source")=="методичка"
                and math.isclose(indirect.get("value",0),expected,abs_tol=.0001)
                and bool(result.get("citations")) and bool(indirect.get("quantity_name")))
        records.append({"name":name,"passed":passed,"result":indirect.get("result_text",result["answer"]),
                        "citations":result.get("citations",[])})
    known=fragments_by_work("1.2")
    records.append({"name":"strict_work_filter","passed":bool(known) and all(row.work=="1.2" for row in known)})
    records.append({"name":"unknown_work_no_cross_work_fallback","passed":not fragments_by_work("999.99")})
    neural=search_manuals("ускорение свободного падения",k=5)
    ranked=retriever.search("ускорение свободного падения")
    records.append({"name":"real_semantic_query_and_reranking","passed":bool(neural) and bool(ranked.hits)
                    and ranked.hits[0].fragment.work=="1.3",
                    "raw_vector_top_work":neural[0].work if neural else None,
                    "adapter_top_work":ranked.hits[0].fragment.work if ranked.hits else None})
    from fastapi.testclient import TestClient
    from app.api.main import app
    response=TestClient(app).post("/calculate",data={"question":"L = 1 м","work":"1.3",
        "mode":"repeat","retrieval_backend":"team","table_text":"T, с\n2\n2.01\n1.99"})
    api_result=response.json()
    records.append({"name":"api_real_team_search","passed":response.status_code==200 and
        api_result.get("payload",{}).get("indirect",{}).get("quantity")=="g"})
    from streamlit.testing.v1 import AppTest
    page=AppTest.from_file(str(ROOT/"src/app/ui/app.py"),default_timeout=30).run()
    page.selectbox[0].set_value("team")
    page.radio[1].set_value("Текст таблицы").run()
    page.text_input[0].set_value("1.3")
    page.text_area[1].set_value("L = 1 м")
    page.button[0].click().run()
    records.append({"name":"ui_real_team_search","passed":len(page.exception)==0 and
        any(widget.value.startswith("**Ускорение свободного падения g = ") for widget in page.markdown)})
    output=ROOT/"artifacts"
    output.mkdir(exist_ok=True)
    report={"checked_utc":datetime.now(timezone.utc).isoformat(),"index":index_manifest(),"checks":records,
            "passed":sum(row["passed"] for row in records),"total":len(records),
            "scope":"Real local MiniLM and Chroma; synthetic methodic PDF. No remote LLM or real photo OCR."}
    (output/"team_search_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=[f"Проверки поисковика + расчётчика: {report['passed']}/{report['total']}"]
    lines += [f"{'PASS' if row['passed'] else 'FAIL'}: {row['name']} — {row.get('result','')}" for row in records]
    text="\n".join(lines)+"\n"
    (output/"team_search_report.txt").write_text(text,encoding="utf-8")
    print(text)
    return int(report["passed"]!=report["total"])
