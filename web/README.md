# Веб-лаборатория `Pi-Predictability-Lab`

Гибридный интерфейс исследования: автономная статическая витрина для GitHub Pages / Vercel + локальный мост **BYO-Compute (Bring Your Own GPU)** на FastAPI.

## Архитектура

- `web/export_static_data.py` — собирает канонический `web/frontend/public/data/lab_data.json` напрямую из `results/01..07/*.csv`, `*.json` и реальных фрагментов из `data/`.
- `web/frontend/` — клиент на Vite + React 19 + TypeScript + Tailwind CSS v4 + Plotly.js + KaTeX + Web Audio API («Пианино $\pi$») + Canvas 2D Random Walk.
- `web/server.py` и `web/services/` — асинхронный локальный мост на `127.0.0.1:8000` (AST-парсер `mpmath` без `eval()`, `PPM-C`, живой инференс PyTorch и многоголовая абляция на чекпойнтах из `runs/`).

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

## 3. Обновление данных витрины из `results/`

После новых прогонов пересоберите `lab_data.json`:

```bash
python web/export_static_data.py
```

## 4. Подключение своей видеокарты к публичному сайту (GitHub Pages)

1. Откройте публичную витрину на `https://w0rsty.github.io/Pi-Predictability-Lab/`.
2. Локально запустите мост: `uvicorn web.server:app --host 127.0.0.1 --port 8000`.
3. В шапке сайта нажмите **«Витрина · Подключить GPU»** и подтвердите `http://127.0.0.1:8000`. Сервер отдаёт заголовки `CORS` и `Access-Control-Allow-Private-Network: true`, поэтому браузер безопасно связывает публичную страницу с вашим локальным процессом без загрузки данных вовне.
