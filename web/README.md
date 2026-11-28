# Веб-лаборатория `Pi-Predictability-Lab`

Гибридный интерфейс исследования: автономная статическая витрина для GitHub Pages / Vercel + локальный мост **BYO-Compute (Bring Your Own GPU)** на FastAPI.

## Архитектура

- `web/export_static_data.py` — собирает канонический `web/frontend/public/data/lab_data.json` напрямую из `results/01..07/*.csv`, `*.json` и реальных фрагментов из `data/`, а также вызывает `export_arena_data.py` и `export_dream_seeds.py`.
- `web/export_onnx_models.py` — экспортирует 4 модели `DigitTransformer` (`true_pi`, `csprng_backdoor`, `champernowne`, `e_cf`) в формат ONNX (opset 17) для автономного инференса в браузере через `onnxruntime-web` (WASM SIMD).
- `web/frontend/` — клиент на Vite + React 19 + TypeScript + Tailwind CSS v4 + Plotly.js + KaTeX + `onnxruntime-web` + Web Audio API («Пианино $\pi$») + Canvas 2D Random Walk.
- `web/server.py` и `web/services/` — асинхронный локальный мост на `127.0.0.1:8000` (AST-парсер `mpmath` без `eval()`, `PPM-C`, живой инференс PyTorch, экспресс-обучение детектора закладок, «Сны нейросети» и многоголовая абляция на чекпойнтах из `runs/`).

## 1. Единый локальный запуск (витрина + GPU-мост)

Установите веб-зависимости и соберите статику один раз:

```bash
pip install -e ".[web]"
cd web/frontend && npm ci && npm run build && cd ../..
uvicorn web.server:app --host 127.0.0.1 --port 8000
```

Откройте `http://127.0.0.1:8000` — мост подключится автоматически.

## 2. Режим разработки (Vite HMR + FastAPI)

В двух терминалах из корня репозитория:

```bash
uvicorn web.server:app --host 127.0.0.1 --port 8000 --reload
```

```bash
cd web/frontend && npm run dev
```

## 3. Обновление данных и ONNX-весов витрины

После новых прогонов пересоберите JSON-артефакты (`lab_data.json`, `arena_rounds.json`, `dream_seeds.json`) и ONNX-модели:

```bash
python web/export_static_data.py
python web/export_onnx_models.py
```

## 4. Подключение своей видеокарты к публичному сайту (GitHub Pages)

1. Откройте публичную витрину на `https://w0rsty.github.io/Pi-Predictability-Lab/`.
2. Локально запустите мост: `uvicorn web.server:app --host 127.0.0.1 --port 8000`.
3. В шапке сайта нажмите кнопку подключения GPU и подтвердите `http://127.0.0.1:8000`. Сервер отдаёт заголовки `CORS` и `Access-Control-Allow-Private-Network: true`, поэтому браузер безопасно связывает публичную страницу с вашим локальным процессом без загрузки данных вовне.

## 5. Демонстрация с чужого ПК через HTTPS-туннель

Чтобы продемонстрировать живое обучение и инференс на домашней RTX 3060 Ti со школьного или чужого ноутбука, защитите мост токеном `LAB_API_TOKEN` и пробросьте порт `8000` через обратный HTTPS-туннель:

1. **Запустите сервер с токеном доступа (дома):**
   ```bash
   LAB_API_TOKEN="ваш-секретный-токен" uvicorn web.server:app --host 127.0.0.1 --port 8000
   ```
2. **Поднимите HTTPS-туннель (выберите один из инструментов):**
   - **Tuna:** `tuna http 8000`
   - **Cloudflare Quick Tunnel:** `cloudflared tunnel --url http://127.0.0.1:8000`
   - **Tailscale Funnel:** `tailscale funnel 8000`
3. **На школьном ноутбуке:** откройте витрину по `https://...`, нажмите на статус `Browser ONNX` в шапке, вставьте `https://`-адрес туннеля и введите токен. Токен передаётся в заголовке `X-Lab-Token` и хранится только в `sessionStorage` текущей вкладки — после её закрытия учётные данные удаляются автоматически.
