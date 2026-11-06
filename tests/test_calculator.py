import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch
import numpy as np
from app.agents.base import AgentContext, DummyTrace
from app.agents.calculator import Calculator
from app.tools import measurements as m
from app.tools.tables import parse_text_table, parse_photo_table, count_numbers
from app.llm.provider import LLMResult, LLMError, FallbackProvider
from app.tools.options import parse_instrument_errors

def calculate(**state):
    return Calculator(AgentContext(DummyTrace())).process(state)

class MeasurementTests(unittest.TestCase):
    def test_direct_95_percent_student(self):
        r = m.process_direct([1, 2, 3, 4, 5])
        self.assertAlmostEqual(r.mean, 3)
        self.assertAlmostEqual(r.standard_error, math.sqrt(.5))
        self.assertAlmostEqual(r.total_error, 2.7764451051977987 * math.sqrt(.5))

    def test_instrument_only_is_expanded(self):
        r = m.process_direct([2]*5, instrument_error=.03)
        self.assertAlmostEqual(r.standard_error, .03/math.sqrt(3))
        self.assertAlmostEqual(r.total_error, 1.95996398454*.03/math.sqrt(3))

    def test_indirect_propagates_standard_uncertainty(self):
        value, error, parts = m.propagate_error("x**2", {"x": 2}, {"x": .1}, {"x": 4})
        self.assertEqual(value, 4)
        self.assertAlmostEqual(parts["x"], .16)
        self.assertAlmostEqual(error, 2.7764451051977987*.4)

    def test_fit_reference(self):
        # Hand solution: x mean=0, Sxx=10, residual SSE=4, df=3.
        r = m.least_squares([-2,-1,0,1,2], [-2,-2,1,2,6])
        self.assertAlmostEqual(r.slope, 2)
        self.assertAlmostEqual(r.intercept, 1)
        residuals = np.array([-2,-2,1,2,6]) - (2*np.array([-2,-1,0,1,2])+1)
        self.assertAlmostEqual(r.slope_standard_error, math.sqrt(float((residuals**2).sum())/3/10))
        self.assertAlmostEqual(r.slope_error, r.slope_standard_error*3.182446305284263)

    def test_fit_constant_x_refused(self):
        with self.assertRaises(ValueError):
            m.least_squares([1,1,1], [2,3,4])

    def test_fit_instrument_floor(self):
        r = m.least_squares([1,2,3], [2,4,6], y_error=.1)
        self.assertGreater(r.slope_standard_error, .07)

    def test_nonfinite_refused(self):
        for values in ([1,float("nan")], [1,float("inf")], []):
            with self.assertRaises(ValueError):
                m.process_direct(values)

    def test_outlier_repeats(self):
        r = m.find_outliers([10,10.01,9.99,10.02,10.01,10,9.98,10,40])
        self.assertEqual(r.outlier_indices, [8])

    def test_fit_outlier(self):
        x = list(range(10))
        y = [2*v+1 for v in x]
        y[4] += 20
        self.assertIn(4, m.fit_outliers(x,y))
        self.assertEqual(m.fit_outliers(x,[2*v+1 for v in x]), [])

class QuantityNameTests(unittest.TestCase):
    def test_names_require_symbol_definition(self):
        from app.tools.quantities import infer_quantity_name
        examples = [
            ("a","Ускорение системы равно удвоенному наклону: a = 1.00 м/с².","Ускорение"),
            ("g","Определить ускорение свободного падения g по периоду.","Ускорение свободного падения"),
            ("E","Коэффициент называется модулем Юнга E.","Модуль Юнга"),
            ("E","где E — ЭДС, r — внутреннее сопротивление.","ЭДС"),
            ("E","Электродвижущая сила E = 4.5 В.","ЭДС"),
            ("eta","eta — коэффициент динамической вязкости.","Коэффициент динамической вязкости"),
            ("rho","Плотность rho = m/V.","Плотность"),
            ("J","Момент инерции маховика составляет J = 0.003 кг·м².","Момент инерции"),
        ]
        for symbol,text,name in examples:
            with self.subTest(symbol=symbol,text=text):
                result=infer_quantity_name(symbol,[text],["методичка, с. 6"])
                self.assertEqual(result.name,name)
                self.assertEqual(result.source,"методичка, с. 6")
                self.assertIn(symbol,result.evidence)

    def test_unknown_or_conflicting_names_not_guessed(self):
        from app.tools.quantities import infer_quantity_name
        for symbol,texts in (("E",["E = x/y"]),
                             ("a",["Ускорение a1 = 1"]),
                             ("E",["Модуль Юнга E = 10.","Энергия E = 5."])):
            self.assertEqual(infer_quantity_name(symbol,texts).name,"")

class TableTests(unittest.TestCase):
    def test_column_display_aliases(self):
        from app.tools.tables import resolve_column_name
        for alias in ("t^2", "t**2", "t²", " t ^ 2 ", "T^2"):
            self.assertEqual(resolve_column_name(alias,["S","t²"]),"t²")
        self.assertEqual(resolve_column_name("s",["S","t²"]),"S")
        self.assertEqual(resolve_column_name("t",["T","t"]),"t")
        with self.assertRaisesRegex(ValueError,"Доступные столбцы"):
            resolve_column_name("x",["S","t²"])
        with self.assertRaisesRegex(ValueError,"неоднозначное"):
            resolve_column_name("s",["S"," S "])
    def test_zero_counted(self):
        self.assertEqual(count_numbers("0 0 1"), 3)

    def test_missing_cell_refused(self):
        for text in ("x; y\n1; 2\n2;\n3; 6", "x; y\n1;2\n2;?\n3;6", "x; y\n1;2;3\n2;4;6"):
            with self.assertRaises(ValueError):
                parse_text_table(text)

    def test_decimal_comma(self):
        table = parse_text_table("x, с; y, м\n1,5; 2,5\n2,5; 3,5")
        self.assertEqual(table.columns, {"x": [1.5,2.5], "y": [2.5,3.5]})

    def test_duplicate_headers_refused(self):
        with self.assertRaises(ValueError):
            parse_text_table("x; x\n1;2\n3;4")

    def photo(self, payload):
        provider = SimpleNamespace(chat=lambda *a, **k: LLMResult(json.dumps(payload)))
        with patch("app.llm.provider.get_provider", return_value=provider):
            return parse_photo_table(b"image")

    def test_photo_decimal_and_zero(self):
        self.assertEqual(self.photo({"columns":{"x":["0","1,5",2]}}).columns["x"], [0,1.5,2])

    def test_photo_force_header_and_lookalike_unit(self):
        for heading in ("F, H", "F, Н", "F"):
            table = self.photo({"columns":{heading:[1,2,3]}, "units":{heading:"H"}})
            self.assertEqual(table.columns,{"F":[1,2,3]})
            self.assertEqual(table.units["F"],"Н")
            self.assertTrue(table.warnings)

    def test_h_not_globally_newton(self):
        from app.tools.units import to_si
        self.assertIsNone(to_si("H"))
        self.assertEqual(self.photo({"columns":{"L":[1,2,3]},"units":{"L":"H"}}).units["L"],"H")
        self.assertEqual(to_si("N"),to_si("Н"))

    def test_squared_time_units(self):
        from app.tools.units import to_si
        for unit in ("с²","с2","с^2","s²","s^2"):
            self.assertEqual(to_si(unit),("с²",1,0))
        self.assertEqual(to_si("мс²"),("с²",1e-6,0))
        self.assertIsNone(to_si("°C²"))

    def test_photo_conflicting_units_and_duplicate_headers(self):
        self.assertTrue(self.photo({"columns":{"L, мм":[1,2,3]},"units":{"L, мм":"м"}}).is_empty)
        self.assertTrue(self.photo({"columns":{"F, Н":[1,2,3],"F":[2,3,4]}}).is_empty)

    def test_photo_malformed(self):
        for payload in ([], {"columns":[]}, {"columns":{"x":[1,None,2]}},
                        {"columns":{"x":[1,2],"y":[3]}}, {"columns":{"x":[True,2]}}):
            self.assertTrue(self.photo(payload).is_empty)

class IntegrationTests(unittest.TestCase):
    def test_ui_import_in_fresh_process(self):
        # Streamlit's script directory must not shadow the app package.
        ui = Path(__file__).resolve().parents[1]/"src/app/ui/app.py"
        code = (
            "import sys,runpy,logging; logging.disable(logging.CRITICAL); "
            f"sys.path.insert(0,{str(ui.parent)!r}); "
            f"runpy.run_path({str(ui)!r},run_name='__main__'); "
            "import app.config; print('IMPORT_OK')"
        )
        result = subprocess.run([sys.executable,"-X","utf8","-c",code], capture_output=True,text=True,encoding="utf-8",timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn("IMPORT_OK",result.stdout)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.patcher = patch.object(m, "settings", SimpleNamespace(artifacts_dir=Path(self.directory.name)))
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_pendulum_assignment_formula(self):
        r = calculate(question="формула g = 4*pi**2*L/T**2\nL = 100 см", table_text="T, с\n2\n2.01\n1.99\n2.02\n1.98", mode="repeat")
        self.assertFalse(r.get("refused"), r["answer"])
        self.assertAlmostEqual(r["payload"]["indirect"]["value"], math.pi**2)
        self.assertEqual(r["payload"]["indirect"]["unit"], "м/с²")

    def test_fit_order_invariant(self):
        rows = [(1,2.1),(2,3.9),(3,6.2),(4,7.8),(5,10.1)]
        def call(rows):
            return calculate(question="формула v = k", mode="fit", x_column="x", y_column="y",
                             table_text="x, с; y, м\n"+"\n".join(f"{x};{y}" for x,y in rows))
        a,b = call(rows),call([rows[i] for i in [3,0,4,1,2]])
        self.assertFalse(a.get("refused"), a["answer"])
        self.assertFalse(b.get("refused"), b["answer"])
        self.assertAlmostEqual(a["payload"]["fit"]["slope"], b["payload"]["fit"]["slope"])
        self.assertNotEqual(a["attachments"], b["attachments"])
        self.assertTrue(Path(a["attachments"][0]).is_file())

    def test_mode_required_for_multiple_columns(self):
        self.assertTrue(calculate(table_text="x;y\n1;2\n2;4\n3;6").get("refused"))

    def test_per_column_units(self):
        r = calculate(question="формула v = L/T", table_text="L, см; T, с\n100;2\n100;2\n100;2", mode="repeat",
                      instrument_errors={"L":{"value":1,"unit":"см"},"T":{"value":.01,"unit":"с"}})
        self.assertFalse(r.get("refused"),r["answer"])
        self.assertAlmostEqual(r["payload"]["direct"]["L"]["standard_error"],.01/math.sqrt(3))
        self.assertAlmostEqual(r["payload"]["direct"]["T"]["standard_error"],.01/math.sqrt(3))
        self.assertEqual(r["payload"]["indirect"]["quantity"],"v")
        self.assertTrue(r["answer"].startswith("v = "))

    def test_wrong_instrument_unit_refused(self):
        r=calculate(table_text="L, м\n1\n1\n1",instrument_errors={"L":{"value":1,"unit":"с"}})
        self.assertTrue(r.get("refused"))

    def test_single_global_instrument_does_not_spread(self):
        r=calculate(question="приборная погрешность 0.01 с", table_text="L, м; T, с\n1;2\n1;2\n1;2",mode="repeat")
        self.assertTrue(r.get("refused"))

    def test_missing_table_no_success(self):
        self.assertTrue(calculate(question="посчитай").get("refused"))

    def test_lookup_without_formula_no_missing_module(self):
        r=calculate(question="Работа 999.9",table_text="T, с\n2\n2.01\n1.99")
        self.assertFalse(r.get("refused"),r["answer"])

    def test_uncertain_x_refused(self):
        r=calculate(mode="fit",table_text="x, с;y, м\n1;2\n2;4\n3;6",instrument_errors={"x":{"value":.1,"unit":"с"}})
        self.assertTrue(r.get("refused"))

    def test_zero_mean_json_serializable(self):
        r=calculate(table_text="x\n0\n0\n0")
        json.dumps(r, allow_nan=False)

    def test_api_text(self):
        from fastapi.testclient import TestClient
        from app.api.main import app
        client=TestClient(app)
        response=client.post("/calculate",data={"table_text":"T, с\n2\n2.01\n1.99","mode":"repeat"})
        self.assertEqual(response.status_code,200,response.text)
        self.assertIn("direct",response.json()["payload"])
        self.assertEqual(client.post("/calculate",data={"mode":"wrong"}).status_code,422)

    def test_options(self):
        self.assertEqual(parse_instrument_errors("T = 0,01 с; L = 1 мм")["T"]["value"],.01)

    def test_streamlit_text_and_fit(self):
        from streamlit.testing.v1 import AppTest
        page = AppTest.from_file(str(Path(__file__).resolve().parents[1]/"src/app/ui/app.py"), default_timeout=20).run()
        self.assertEqual(len(page.exception),0)
        page.radio[1].set_value("Текст таблицы").run()
        page.button[0].click().run()
        self.assertEqual(len(page.exception),0)
        self.assertEqual(len(page.success),1)
        page.radio[0].set_value("Зависимость y(x), МНК").run()
        page.button[0].click().run()
        self.assertEqual(len(page.exception),0)
        self.assertEqual(len(page.success),1)

    def test_photo_and_text_use_same_methodic_formula(self):
        # The model supplies only transcription; the real local PDF index
        # supplies the formula and citations for both input paths.
        common = {"question":"L = 1 м", "work":"1.3", "mode":"repeat"}
        text = calculate(**common, table_text="T, с\n2\n2.01\n1.99\n2.02\n1.98")
        provider = SimpleNamespace(chat=lambda *a, **kw: LLMResult(json.dumps({
            "columns":{"T":[2,2.01,1.99,2.02,1.98]},"units":{"T":"с"}})))
        with patch("app.llm.provider.get_provider",return_value=provider):
            photo = calculate(**common,image=b"photo")
        for result in (text,photo):
            self.assertFalse(result.get("refused"),result["answer"])
            self.assertEqual(result["payload"]["indirect"]["source"],"методичка")
            self.assertTrue(result["citations"])
            self.assertTrue(result["payload"]["methodics"])
            self.assertAlmostEqual(result["payload"]["indirect"]["value"],math.pi**2)
        self.assertEqual(text["payload"]["indirect"]["value"],photo["payload"]["indirect"]["value"])

    def test_photo_api_uses_work_field(self):
        from fastapi.testclient import TestClient
        from app.api.main import app
        provider = SimpleNamespace(chat=lambda *a, **kw: LLMResult(json.dumps({
            "columns":{"T":[2,2.01,1.99]},"units":{"T":"с"}})))
        with patch("app.llm.provider.get_provider",return_value=provider):
            response = TestClient(app).post("/calculate",data={"work":"1.3","question":"L = 1 м"},
                files={"image":("table.png",b"photo","image/png")})
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(response.json()["payload"]["indirect"]["source"],"методичка")

    def test_atwood_photo_square_time_and_acceleration(self):
        payload={"columns":{"S, м":[.2,.4,.6,.8,1.0],"t², с²":[.41,.8,1.22,1.59,2.01]},
                 "units":{"S, м":"м","t², с²":"с²"}}
        provider=SimpleNamespace(chat=lambda *a, **kw: LLMResult(json.dumps(payload)))
        with patch("app.llm.provider.get_provider",return_value=provider):
            result=calculate(image=b"photo",question="обработай измерения",work="1.2",mode="fit",x_column="t^2",y_column="s")
        self.assertFalse(result.get("refused"),result["answer"])
        indirect=result["payload"]["indirect"]
        self.assertEqual(indirect["source"],"методичка")
        self.assertEqual(indirect["formula"],"2*k")
        self.assertEqual(indirect["unit"],"м/с²")
        self.assertAlmostEqual(indirect["value"],1.0,places=2)
        self.assertEqual(indirect["unit_matches_methodics"],True)
        self.assertEqual(indirect["quantity"],"a")
        self.assertEqual(indirect["quantity_name"],"Ускорение")
        self.assertTrue(result["answer"].startswith("Ускорение a = "))
        self.assertIn("Ускорение",indirect["quantity_name_evidence"])
        self.assertIn("с. 6",indirect["quantity_name_source"])
        self.assertTrue(result["attachments"])
        self.assertEqual(result["payload"]["fit"]["x_column"],"t²")
        self.assertEqual(result["payload"]["fit"]["y_column"],"S")

    def test_named_result_in_ui(self):
        from streamlit.testing.v1 import AppTest
        page=AppTest.from_file(str(Path(__file__).resolve().parents[1]/"src/app/ui/app.py"),default_timeout=20).run()
        page.radio[1].set_value("Текст таблицы").run()
        page.text_input[0].set_value("1.3")
        page.text_area[1].set_value("L = 1 м")
        page.button[0].click().run()
        self.assertEqual(len(page.exception),0)
        self.assertTrue(any(widget.value.startswith("**Ускорение свободного падения g = ") for widget in page.markdown))

if __name__ == "__main__":
    unittest.main()
