"""Run: .\\start.ps1 [ui|up|api|setup|index|test|eval|demo|provider|check]."""
from pathlib import Path
import json
import os
import subprocess
import sys

# Windows console encodings may not contain symbols such as м/с² and ±.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
       "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT),
       "MPLCONFIGDIR": str(ROOT / "artifacts/matplotlib-cache")}
os.environ.setdefault("MPLCONFIGDIR", ENV["MPLCONFIGDIR"])
os.chdir(ROOT)

def run(args, env=None):
    return subprocess.call([sys.executable, "-X", "utf8", *args], cwd=ROOT, env=env or ENV)

def setup():
    from importlib.metadata import version, PackageNotFoundError
    missing = []
    for raw in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, expected = line.split("==", 1)
        try:
            installed = version(name)
        except PackageNotFoundError:
            installed = "not installed"
        if installed != expected:
            missing.append(f"{name}: {installed}, expected {expected}")
    if missing:
        print("Dependencies differ:\n" + "\n".join(missing))
        print("Install with a working Python: python -m pip install -r requirements.txt")
        return 1
    if not (ROOT / ".env").exists():
        (ROOT / ".env").write_text((ROOT / ".env.example").read_text(encoding="utf-8"), encoding="utf-8")
    from app.data.index import build_index
    if not list((ROOT / "data/methodics").glob("*.pdf")):
        print("Add methodic PDFs to data/methodics; text + explicit formulas work without an index.")
    else:
        print(json.dumps(build_index(backend="bm25"), ensure_ascii=False, indent=2))
    return 0

def main():
    command = sys.argv[1] if len(sys.argv) > 1 else "ui"
    if command in {"search-model", "search-ocr-model", "search-parse", "search-index", "search-test"}:
        from app.team_search.cli import execute
        return execute(command)
    if command == "setup":
        return setup()
    if command == "index":
        from app.data.index import build_index
        print(json.dumps(build_index(), ensure_ascii=False, indent=2))
        return 0
    if command == "test":
        # Numerical regressions must never call the configured remote model.
        return run(["-m", "unittest", "discover", "-s", "tests", "-v"], {**ENV, "LLM_PROVIDER": "mock", "RETRIEVAL_BACKEND": "legacy"})
    if command == "eval":
        return run(["eval/run_eval.py"], {**ENV, "LLM_PROVIDER": "mock", "RETRIEVAL_BACKEND": "legacy"})
    if command == "demo":
        from app.agents.calculator import Calculator
        from app.agents.base import AgentContext, DummyTrace
        sample = {"question": "формула g = 4*pi**2*L/T**2\nL = 1 м",
                  "table_text": "T, с\n2.00\n2.01\n1.99\n2.02\n1.98", "mode": "repeat",
                  "instrument_errors": {"T": {"value": 0.01, "unit": "с"}}}
        result = Calculator(AgentContext(DummyTrace())).process(sample)
        print(result["answer"])
        return int(bool(result.get("refused")))
    if command == "provider":
        from app.config import settings
        from app.llm.provider import get_provider
        print(f"Provider={settings.provider}; text={settings.model}; vision={settings.vision_model}")
        try:
            result = get_provider().chat([{"role": "user", "content": "Ответь одним словом: готов"}])
            print(f"Model={result.model}; answer={result.text}")
            return 0
        except Exception as exc:
            print(str(exc))
            return 1
    if command == "check":
        for step in ("test", "eval"):
            code = run(["run.py", step])
            if code:
                return code
        return 0
    api = ["-m", "uvicorn", "app.api.main:app", "--host", "127.0.0.1", "--port", "8000"]
    ui = ["-m", "streamlit", "run", "src/app/ui/app.py", "--server.address", "127.0.0.1", "--server.port", "8501"]
    if command in {"api", "ui"}:
        return run(api if command == "api" else ui)
    if command == "up":
        process = subprocess.Popen([sys.executable, "-X", "utf8", *api], cwd=ROOT, env=ENV)
        try:
            return run(ui)
        finally:
            process.terminate()
            process.wait(timeout=10)
    print(__doc__)
    return 2

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as exc:
        print(str(exc))
        raise SystemExit(1)
    except KeyboardInterrupt:
        raise SystemExit(130)
