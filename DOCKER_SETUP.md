# Running PDF Agent with Docker

No Python installation needed — Docker handles everything.

---

## Step 1 — Install Docker Desktop

1. Go to https://www.docker.com/products/docker-desktop/
2. Download the **Windows** installer and run it
3. After installation, open **Docker Desktop** from your Start menu
4. Wait until you see the green "Docker is running" status in the system tray

> You only need to do this once.

---

## Step 2 — Set up your API Key

The app needs an LLM API key to classify and summarize documents.

1. In the `pdf-agent` folder, find the file called `.env.example`
2. Make a copy of it and name the copy `.env` (no `.example`)
3. Open `.env` in Notepad and fill in your key:

```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-actual-key-here
```

> If you're using Anthropic Claude instead, set:
> `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY=sk-ant-...`

Save and close the file.

---

## Step 3 — Open a Terminal in the project folder

1. Open **File Explorer** and navigate to the `pdf-agent` folder
2. Click the address bar at the top, type `cmd`, and press Enter
3. A black Command Prompt window opens already inside that folder

---

## Step 4 — Build and start the app

Run this single command:

```
docker compose up --build
```

**What this does:**
- `--build` — builds the Docker images the first time (downloads Python, installs all packages)
- This takes **5–10 minutes** the first time because it downloads the embedding model
- After the first run, subsequent starts take only a few seconds

You'll see lots of output scrolling by. Wait until you see something like:

```
pdf-agent-backend   | INFO: Application startup complete.
pdf-agent-frontend  | You can now view your Streamlit app in your browser.
```

---

## Step 5 — Open the app

Open your browser and go to:

**http://localhost:8501**

That's the Streamlit UI. You can also open **http://localhost:8000/docs** to see the raw API.

---

## Everyday usage

| What you want to do | Command |
|---|---|
| Start the app | `docker compose up` |
| Start in background (no terminal needed) | `docker compose up -d` |
| Stop the app | `docker compose down` |
| See logs | `docker compose logs -f` |
| Rebuild after code changes | `docker compose up --build` |

---

## Common errors and fixes

### "Cannot connect to the Docker daemon"
Docker Desktop isn't running. Open it from the Start menu and wait for it to turn green.

### "port is already allocated"
Something else is using port 8000 or 8501. Stop the other app, or change the ports in `docker-compose.yml`:
```yaml
ports:
  - "8080:8000"   # Use 8080 instead of 8000
```

### "Backend offline" shown in the Streamlit sidebar
The backend is still starting up. Wait 30 seconds and refresh the page.

### The embedding model download is slow
The BAAI/bge-small-en-v1.5 model (~130 MB) downloads once on first build. Subsequent starts use the cached version.

### I changed my API key
Edit `.env`, then run `docker compose up` (no `--build` needed — `.env` is read at startup, not baked into the image).

---

## Folder structure reminder

```
pdf-agent/
├── .env                  ← Your API keys (you create this)
├── .env.example          ← Template (don't edit this)
├── docker-compose.yml    ← Defines the two services
├── Dockerfile            ← Backend image
├── Dockerfile.frontend   ← Frontend image
├── data/
│   ├── uploads/          ← Your uploaded PDFs are stored here
│   └── vectorstore/      ← ChromaDB embeddings stored here
└── ...
```

The `data/` folder is mounted as a **volume**, meaning your uploaded PDFs and their embeddings survive when you stop and restart Docker.
