Prerequisites

- docker (https://docs.docker.com/engine/install/)
- uv (https://docs.astral.sh/uv/getting-started/installation/)
- groq api key (https://console.groq.com/keys)

How to run

```bash
git clone https://github.com/yatin-ys/algorithmx-assignment.git

uv sync # for python virtual env

source .venv/bin/activate # activate env

docker compose up -d #for postgres and qdrant

python -m backend.db.migrate # run migration, make sure postgres is running 

fastapi dev backend/api/main.py # for fastapi server

streamlit run ui/app.py # for streamlit ui
```