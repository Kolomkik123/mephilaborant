"""Streamlit interface for repeat measurements and linear fits."""
import json
import sys
from pathlib import Path
# Streamlit puts this script's directory first. Resolve the app package from
# src before importing it, otherwise this app.py shadows the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import streamlit as st
from app.config import settings
from app.data.index import index_manifest
from app.agents.calculator import Calculator
from app.agents.base import AgentContext, DummyTrace
from app.tools.tables import parse_text_table
from app.tools.options import parse_instrument_errors

st.set_page_config(page_title="Расчётчик МИФИ", page_icon="🧮")
st.title("🧮 Расчётчик измерений")
st.write("Загрузите фото таблицы или введите измерения текстом. По номеру работы расчётчик найдёт формулу и методику в PDF.")
with st.sidebar:
    st.caption(f"Провайдер: {settings.provider}")
    st.caption(f"Модель текста: {settings.model}; зрения: {settings.vision_model}")
    st.metric("Фрагментов методички", index_manifest().get("fragments", 0))
    if settings.is_mock:
        st.info("Mock: расчёты по тексту работают без сети. Для фото нужна модель зрения.")

mode_label = st.radio("Что обрабатываем", ["Повторные измерения", "Зависимость y(x), МНК"])
mode = "repeat" if mode_label == "Повторные измерения" else "fit"
source = st.radio("Данные", ["Фотография", "Текст таблицы"], horizontal=True, key="data_source")
example = "T, с\n2.00\n2.01\n1.99\n2.02\n1.98" if mode == "repeat" else "x, с; y, м\n1; 2.1\n2; 3.9\n3; 6.2\n4; 7.8\n5; 10.1"
table_text, photo = "", None
names = []
if source == "Текст таблицы":
    table_text = st.text_area("Таблица с заголовком и единицами", example, height=180, key=f"table_{mode}")
    st.caption("Разделяйте столбцы точкой с запятой или табуляцией. Пропуски нужно исправить.")
    try:
        names = list(parse_text_table(table_text).columns)
    except ValueError as exc:
        st.warning(str(exc))
else:
    photo = st.file_uploader("Фото таблицы", type=["png", "jpg", "jpeg"])
    st.caption("Числа после распознавания будут показаны для сверки с журналом.")
work = st.text_input("Номер лабораторной работы", placeholder="Например: 1.3",
    help="Для поиска формулы и теории в методичке из data/methodics. Работает и с фото, и с текстовой таблицей.")
question = st.text_area("Что найти и какие величины заданы (необязательно)",
    "",
    placeholder="Например: найди ускорение свободного падения. L = 1 м",
    help="Формулу можно оставить пустой: расчётчик найдёт её в методичке выбранной работы. Если знаете формулу, напишите: формула g = 4*pi**2*L/T**2",
    key=f"question_{mode}")
errors_text = st.text_input("Приборные границы погрешностей", placeholder="T = 0.01 с; L = 1 мм",
    help="Укажите каждую величину отдельно. Пропущенная величина считается без приборной составляющей.")
x_column = y_column = ""
if mode == "fit":
    if len(names) >= 2:
        x_column = st.selectbox("Ось x", names)
        y_column = st.selectbox("Ось y", [n for n in names if n != x_column])
    else:
        x_column = st.text_input("Обозначение x на фото")
        y_column = st.text_input("Обозначение y на фото")
    st.caption("Линейный МНК: x считается точным. Подозрительные точки отмечаются без удаления.")
    st.caption("Оси выбирайте по методичке. Например, в образце работы 1.2: x = t², y = S; ускорение находится из наклона графика.")
if st.button("Посчитать", type="primary"):
    try:
        state = {"question": question, "work": work.strip(), "table_text": table_text, "mode": mode,
                 "instrument_errors": parse_instrument_errors(errors_text),
                 "x_column": x_column, "y_column": y_column}
        if source == "Фотография":
            if photo is None:
                raise ValueError("Прикрепите фотографию или выберите текстовый ввод")
            state["image"] = photo.getvalue()
        with st.spinner("Распознаю таблицу, ищу формулу в методичке и считаю…" if source == "Фотография" else "Обрабатываю таблицу и ищу формулу в методичке…"):
            result = Calculator(AgentContext(DummyTrace())).process(state)
        if result.get("refused"):
            st.warning(result["answer"])
        else:
            st.success("Расчёт завершён")
            summary = result.get("payload", {}).get("indirect", {}).get("result_text", "")
            if summary:
                st.markdown(f"**{summary}**")
                with st.expander("Ход расчёта и погрешности"):
                    st.markdown(result["answer"])
            else:
                st.markdown(result["answer"])
            if result.get("payload", {}).get("columns"):
                st.write("Числа, использованные в расчёте (СИ)")
                st.dataframe(result["payload"]["columns"])
            for path in result.get("attachments", []):
                if Path(path).is_file():
                    st.image(path)
            if result.get("citations"):
                st.caption("Источник: " + "; ".join(result["citations"]))
            methodics = result.get("payload", {}).get("methodics", [])
            if methodics:
                with st.expander("Теория и методика из PDF"):
                    for fragment in methodics:
                        st.caption(fragment["citation"])
                        st.text(fragment["text"])
            st.download_button("Скачать результат JSON", json.dumps(result, ensure_ascii=False, indent=2),
                               file_name="calculation.json", mime="application/json")
    except ValueError as exc:
        st.warning(str(exc))
