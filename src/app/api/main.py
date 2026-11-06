"""Calculator API; forms also accept plain text without an image."""
from typing import Any, Literal
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from starlette.concurrency import run_in_threadpool
from app.config import settings
from app.data.index import build_index, index_manifest
from app.agents.calculator import Calculator
from app.agents.base import AgentContext, DummyTrace
from app.tools.options import parse_instrument_errors

app = FastAPI(title="MIFI Calculator API", version="1.1.0")

@app.get("/health")
def health() -> dict[str, Any]:
    manifest = index_manifest()
    return {"status": "ok", "provider": settings.provider, "index": manifest}

@app.post("/calculate")
async def calculate(question: str = Form(""), table_text: str = Form(""),
                    work: str = Form(""),
                    mode: Literal["repeat", "fit"] = Form("repeat"),
                    x_column: str = Form(""), y_column: str = Form(""),
                    instrument_errors: str = Form(""), student_id: str = Form("anon"),
                    image: UploadFile | None = File(None)) -> dict[str, Any]:
    try:
        errors = parse_instrument_errors(instrument_errors)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    state = {"question": question, "work": work.strip(), "table_text": table_text, "mode": mode,
             "x_column": x_column, "y_column": y_column, "instrument_errors": errors}
    if image:
        state["image"] = await image.read()
    calculator = Calculator(AgentContext(DummyTrace(), student_id=student_id))
    return await run_in_threadpool(calculator.process, state)

@app.post("/index/rebuild")
def rebuild_index() -> dict[str, Any]:
    return build_index()
