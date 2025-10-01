How to run

```bash
git clone https://github.com/yatin-ys/algorithmx-assignment.git

uv sync # for python virtual env

docker compose up -d #for postgres and qdrant

fastapi dev backend/api/main.py # for fastapi server

streamlit run ui/app.py # for streamlit ui
```