# Evaluation Benchmark — начальный план

Размещайте тесты в JSONL/JSON с полями:
- `test_id`
- `category`
- `input`
- `expected`
- `rationale`

Стенд должен сравнивать предсказание Проверяющего с эталонной разметкой.
Минимальные метрики: precision, recall, F1, false acceptance rate, false rejection rate,
latency, token usage, retry rate.

Не смешивайте `needs_review` с `pass`: это отдельный исход, требующий анализа.
Для сравнения системы с Checker и без Checker используйте одинаковые входные кейсы,
одинаковые версии моделей и фиксируйте параметры запуска.
