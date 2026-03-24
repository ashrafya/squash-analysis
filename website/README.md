# Squash Analysis Website

## Setup

### 1. Install Node.js (if not already)
```bash
# macOS with Homebrew
brew install node

# Or via nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
nvm install 20
```

### 2. Install frontend dependencies
```bash
cd website
npm install
```

### 3. Install API dependencies
```bash
# From the project root, activate your venv first
source ../venv/bin/activate
pip install -r api/requirements.txt
```

## Running

Open two terminals:

**Terminal 1 — FastAPI backend**
```bash
cd website/api
source ../../venv/bin/activate
uvicorn server:app --reload --port 8000
```

**Terminal 2 — Next.js frontend**
```bash
cd website
npm run dev
```

Then open http://localhost:3000

## Notes
- The API wraps the existing Python pipeline in `src/`
- Jobs run sequentially (one at a time for now)
- Court calibration must be done once: `cd src && python main.py --calibrate`
- Output images are served from `website/api/jobs/<job_id>/`
