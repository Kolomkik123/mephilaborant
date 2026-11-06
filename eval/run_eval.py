"""Small synthetic reference suite, not a real-lab benchmark."""
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from app.tools import measurements as m

root=Path(__file__).resolve().parents[1]
cases=json.loads((root/"eval/calculations.json").read_text(encoding="utf-8"))
results=[]
for case in cases:
    kind=case["kind"]
    try:
        if kind=="direct":
            result=m.process_direct(case["values"], instrument_error=case.get("instrument_error",0))
            actual={"mean": result.mean,"error":result.total_error}
        elif kind=="fit":
            result=m.least_squares(case["x"],case["y"])
            actual={"slope":result.slope,"intercept":result.intercept}
        elif kind=="indirect":
            value,error,_=m.propagate_error(case["formula"],case["values"],case["errors"])
            actual={"value":value,"error":error}
        elif kind=="outlier":
            actual={"indices":m.find_outliers(case["values"]).outlier_indices}
        else:
            actual={"indices":m.fit_outliers(case["x"],case["y"])}
        passed=all(actual[k]==case[k] if isinstance(case[k],list) else math.isclose(actual[k],case[k],rel_tol=1e-7,abs_tol=1e-10)
                   for k in actual if k in case)
        results.append({"id":case["id"],"passed":passed,"actual":actual})
    except Exception as exc:
        results.append({"id":case["id"],"passed":False,"error":str(exc)})
out=root/"artifacts"
out.mkdir(exist_ok=True)
passed=sum(row["passed"] for row in results)
report={"time_utc":datetime.now(timezone.utc).isoformat(),"dataset":"synthetic references v1",
        "provider":"none: deterministic Python tools only","passed":passed,"total":len(results),"results":results}
(out/"metrics.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
text=f"# Проверка синтетических эталонов\n\nПройдено: {passed}/{len(results)}.\n\nЭто малый проверочный набор, не доказательство ≥95% на реальных лабораторных работах или ≥90% полноты поиска промахов. OCR и качество моделей здесь не измеряются.\n\n"
text+="\n".join(f"- {row['id']}: {'PASS' if row['passed'] else 'FAIL'}" for row in results)
(out/"metrics.md").write_text(text+"\n",encoding="utf-8")
print(text)
raise SystemExit(0 if passed==len(results) else 1)
