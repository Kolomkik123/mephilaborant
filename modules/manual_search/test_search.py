from tools.search import search_manuals

print("=== ТЕСТ 1: Поиск без фильтра по работе ===")
query_1 = "Как найти кинетическую энергию?"
results_1 = search_manuals(query_1, k=3)

print(f"Найдено результатов: {len(results_1)}")

if not results_1:
    print("Ничего не найдено. Скорее всего, порог SIMILARITY_THRESHOLD слишком высокий. Понизь его в search.py до -1.0")
else:
    for i, r in enumerate(results_1):
        print(f"\n--- Источник [{i+1}] ---")
        print(f"Работа: {r.work}, Страница: {r.page}, Близость: {r.similarity}")
        print(f"Текст: {r.text[:200].replace(chr(10), ' ')}...")

print("\n\n=== ТЕСТ 2: Поиск с фильтром по работе (16) ===")
query_2 = "Что такое центр качания маятника?"
results_2 = search_manuals(query_2, work="16", k=2)

print(f"Найдено результатов: {len(results_2)}")
for i, r in enumerate(results_2):
    print(f"\n--- Источник [{i+1}] ---")
    print(f"Работа: {r.work}, Страница: {r.page}, Близость: {r.similarity}")
    print(f"Текст: {r.text[:200].replace(chr(10), ' ')}...")