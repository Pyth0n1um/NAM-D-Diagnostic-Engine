# app.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ingest import ingest, ValidationError
from Pipeline.execute import run_pipeline

import inspect

app = FastAPI(title="NAM-D Diagnostic Engine API")

print(">>> app.py loaded <<<")
print(">>> APP OBJECT ID:", id(app))
print(">>> APP DEFINED IN:", inspect.getfile(app.__class__))

try:
    from Pipeline.execute import run_pipeline
    print(">>> run_pipeline imported OK <<<")
except Exception as e:
    print(">>> run_pipeline import FAILED:", e)

try:
    from ingest import ingest
    print(">>> ingest imported OK <<<")
except Exception as e:
    print(">>> ingest import FAILED:", e)


# ---------------------------------------------------------
# CORS (adjust as needed for your front-end)
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print(">>> Initialized app.py from NAM-D_App <<<")


# ---------------------------------------------------------
# REST ENDPOINT (simple synchronous analysis)
# ---------------------------------------------------------
@app.post("/analyze")
async def analyze(payload: dict):
    try:
        clean = ingest(payload)
        result = run_pipeline(clean["narrative_text"], clean["ta_raw"])

        # Convert plain Python object → JSON-serializable dict
        return JSONResponse(content=result.__dict__)

    except ValidationError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    except Exception as e:
        import traceback
        print(">>> INTERNAL SERVER ERROR:")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
