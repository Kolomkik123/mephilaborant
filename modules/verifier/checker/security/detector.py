import re
from checker.schemas import SecurityResult

# Heuristic patterns only. Keep this list versioned and evaluate false positives.
PATTERNS = {
    # 1. Игнорирование предыдущих инструкций
    "ignore_previous_instructions": (
        r"(ignore|disregard|forget|discard|игнорируй|игнорировать|"
        r"проигнорируй|забудь|отбрось|не\s+учитывай)"
        r".{0,60}"
        r"(previous|prior|above|earlier|предыдущ\w*|"
        r"предшествующ\w*|вышеуказанн\w*|прошл\w*)"
        r".{0,30}"
        r"(instructions?|prompts?|rules?|инструкци\w*|"
        r"указани\w*|правил\w*|сообщени\w*)"
    ),

    # 2. Игнорирование системных инструкций
    "ignore_system_instructions": (
        r"(ignore|disregard|forget|игнорируй|забудь|"
        r"проигнорируй|не\s+следуй)"
        r".{0,50}"
        r"(system|developer|системн\w*|разработчик\w*)"
        r".{0,30}"
        r"(instructions?|prompt|инструкци\w*|промпт\w*)"
    ),

    # 3. Раскрытие системного промпта
    "reveal_system_prompt": (
        r"(reveal|show|print|display|output|выведи|покажи|"
        r"раскрой|напечатай|процитируй|скопируй)"
        r".{0,60}"
        r"(system\s*prompt|hidden\s*prompt|"
        r"системн\w*\s+промпт|скрыт\w*\s+инструкци\w*|"
        r"системн\w*\s+инструкци\w*)"
    ),

    # 4. Обход ограничений и правил
    "override_rules": (
        r"(ignore|bypass|override|disable|break|обойди|"
        r"отмени|нарушь|отключи|сними|игнорируй|"
        r"отмени|переопредели)"
        r".{0,60}"
        r"(rules?|restrictions?|limitations?|polic(?:y|ies)|"
        r"правил\w*|ограничени\w*|политик\w*|"
        r"защит\w*|фильтр\w*)"
    ),

    # 5. Попытка изменить роль агента
    "role_hijack": (
        r"(you\s+are\s+now|act\s+as|pretend\s+to\s+be|"
        r"from\s+now\s+on|теперь\s+ты|с\s+этого\s+момента\s+ты|"
        r"представь\s+что\s+ты|действуй\s+как|"
        r"выступай\s+в\s+роли)"
        r".{0,60}"
        r"(system|developer|admin|administrator|root|"
        r"администратор|разработчик|системн\w*|"
        r"неограниченн\w*|без\s+ограничений)"
    ),

    # 6. Попытка получить секретные данные
    "exfiltrate_secrets": (
        r"(api[_ -]?key|access[_ -]?token|password|secret|"
        r"credentials|парол\w*|ключ\w*\s+доступа|"
        r"секретн\w*\s+ключ\w*|токен\w*|"
        r"учетн\w*\s+данн\w*)"
    ),

    # 7. Попытка получить внутренние данные
    "reveal_internal_data": (
        r"(show|reveal|print|output|выведи|покажи|"
        r"раскрой|отправь|предоставь)"
        r".{0,60}"
        r"(internal\s+data|hidden\s+data|"
        r"внутренн\w*\s+данн\w*|"
        r"скрыт\w*\s+данн\w*|"
        r"внутренн\w*\s+состояни\w*|"
        r"содержимое\s+state)"
    ),

    # 8. Попытка выполнить команды вместо обычной задачи
    "instruction_override": (
        r"(instead\s+of\s+the\s+original\s+task|"
        r"вместо\s+исходн\w*\s+задач\w*|"
        r"выполняй\s+следующ\w*\s+инструкци\w*|"
        r"теперь\s+твоя\s+задача|"
        r"новая\s+главная\s+задача|"
        r"забудь\s+о\s+своей\s+задаче)"
    ),

    # 9. Попытка активировать режим без ограничений
    "jailbreak": (
        r"(developer\s+mode|dan\s+mode|jailbreak|"
        r"режим\s+разработчика|режим\s+без\s+ограничений|"
        r"режим\s+свободного\s+ответа|"
        r"отвечай\s+без\s+ограничений|"
        r"отключи\s+все\s+ограничения)"
    ),

    # 10. Попытка подменить источник инструкций
    "fake_system_message": (
        r"(system|developer|assistant)\s*:"
        r"|"
        r"(системное\s+сообщение|сообщение\s+разработчика)"
        r".{0,40}"
        r"(новые\s+инструкции|new\s+instructions|"
        r"приоритетная\s+инструкция)"
    ),

    # 11. Попытка заставить модель раскрыть ход рассуждений
    "reveal_chain_of_thought": (
        r"(show|reveal|print|output|покажи|выведи|"
        r"раскрой|напиши)"
        r".{0,60}"
        r"(chain\s+of\s+thought|internal\s+reasoning|"
        r"скрыт\w*\s+рассуждени\w*|"
        r"внутренн\w*\s+рассуждени\w*|"
        r"внутренн\w*\s+мысл\w*)"
    ),
}


def normalize_text(text: str) -> str:
    text = text.lower()

    # Унифицируем русские буквы
    text = text.replace("ё", "е")

    # Убираем лишние пробелы и переносы строк
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def detect_injection(text: str) -> SecurityResult:
    text = normalize_text(text) or ""
    matches = [
        name for name, pattern in PATTERNS.items()
        if re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    ]
    suspicious = bool(matches)
    return SecurityResult(
        suspicious=suspicious,
        matched_patterns=matches,
        action="isolate" if suspicious else "allow",
        explanation=(
            "Обнаружены подозрительные формулировки. Содержимое следует трактовать "
            "как недоверенные данные, а не исполнять как инструкции."
            if suspicious else "Типичные эвристические признаки инъекции не обнаружены."
        ),
    )
