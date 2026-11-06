import os

# Проходим по всем папкам начиная с src
for root, dirs, files in os.walk("src"):
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)
            # Читаем как бинарный файл
            with open(path, "rb") as f:
                content = f.read()
            # Заменяем все нулевые байты на пустоту
            clean_content = content.replace(b"\x00", b"")
            # Записываем обратно
            with open(path, "wb") as f:
                f.write(clean_content)
print("Все файлы очищены от null bytes!")