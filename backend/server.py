"""Local RallyIQ analysis API. Run: uvicorn backend.server:app --reload"""
from __future__ import annotations
import json, os, uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from .analyzer import analyze_video
from .sabermetrics import summarize
from .partnerships import summarize_partnerships
from .adjusted_plus_minus import fit_apm
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'analysis_data'
DATA.mkdir(exist_ok=True)
app = FastAPI(title='RallyIQ Video Analysis', version='0.1.0')
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:8000','http://127.0.0.1:8000'], allow_credentials=False, allow_methods=['GET','POST'], allow_headers=['*'])
executor = ThreadPoolExecutor(max_workers=2)
jobs = {}
lock = Lock()
MAX_BYTES = 1024 * 1024 * 1024

@app.get('/')
def index():
    return FileResponse(ROOT / 'index.html')

@app.get('/api/health')
def health():
    return {'status':'ok', 'analysis':'motion-based candidate rally detection'}

def process(job_id, path):
    with lock: jobs[job_id]['status'] = 'processing'
    try:
        result = analyze_video(str(path))
        (DATA / (job_id+'.json')).write_text(json.dumps(result, indent=2))
        with lock: jobs[job_id].update(status='complete', result=result)
    except Exception as e:
        with lock: jobs[job_id].update(status='failed', error=str(e))

@app.post('/api/analyze', status_code=202)
async def upload_video(file: UploadFile = File(...)):
    ext = Path(file.filename or '').suffix.lower()
    if ext not in {'.mp4','.mov','.m4v','.avi','.webm'}:
        raise HTTPException(400,'Unsupported file extension. Use MP4, MOV, M4V, AVI or WEBM.')
    job_id = uuid.uuid4().hex
    path = DATA / (job_id+ext)
    size = 0
    try:
        with path.open('wb') as out:
            while chunk := await file.read(1024*1024):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(413,'Video exceeds 1 GB limit.')
                out.write(chunk)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    with lock: jobs[job_id] = {'id':job_id,'status':'queued','filename':file.filename}
    executor.submit(process, job_id, path)
    return {'job_id':job_id,'status':'queued'}

@app.get('/api/jobs/{job_id}')
def get_job(job_id: str):
    with lock: job = jobs.get(job_id)
    if not job:
        saved = DATA / (job_id+'.json')
        if saved.exists():
            return {'id':job_id,'status':'complete','result':json.loads(saved.read_text())}
        raise HTTPException(404,'Job not found')
    return job

class RallyRecord(BaseModel):
    player: str = Field(min_length=1, max_length=120)
    partner: str = ''
    opponent: str = ''
    won: bool
    match_id: str = ''
    rally_id: str = ''

class SabermetricsRequest(BaseModel):
    rallies: list[RallyRecord] = Field(max_length=100000)
    replacement_rate: float = Field(default=0.40, ge=0, le=1)
    points_per_win: float = Field(default=11, gt=0, le=100)

@app.post('/api/sabermetrics')
def calculate_sabermetrics(payload: SabermetricsRequest):
    return summarize([r.model_dump() for r in payload.rallies],
                     replacement_rate=payload.replacement_rate,
                     points_per_win=payload.points_per_win)

@app.post('/api/partnerships')
def calculate_partnerships(payload: SabermetricsRequest):
    return summarize_partnerships([r.model_dump() for r in payload.rallies],
                                  replacement_rate=payload.replacement_rate,
                                  points_per_win=payload.points_per_win)

class APMRecord(RallyRecord):
    opponent2: str = Field(min_length=1, max_length=120)
    serve_team: str

class APMRequest(BaseModel):
    rallies: list[APMRecord] = Field(min_length=20, max_length=100000)
    penalty: float = Field(default=25.0, ge=1, le=1000)
    pair_penalty: float = Field(default=80.0, ge=1, le=1000)

@app.post('/api/adjusted-plus-minus')
def adjusted_plus_minus(payload: APMRequest):
    try:
        return fit_apm([r.model_dump() for r in payload.rallies], payload.penalty, payload.pair_penalty)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

from .pwar import pwar

class PWARRequest(APMRequest):
    ratings: dict[str, float]
    target: int = 11

@app.post('/api/pwar')
def calculate_pwar(payload: PWARRequest):
    try:
        return pwar([r.model_dump() for r in payload.rallies], payload.ratings, payload.target)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

from .partnership_pwar import partnership_pwar

@app.post('/api/partnership-pwar')
def calculate_partnership_pwar(payload: PWARRequest):
    try:
        return partnership_pwar([r.model_dump() for r in payload.rallies], payload.ratings, payload.target)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

from .rally_records import build_records
from typing import Literal

class VerifiedLabel(BaseModel):
    rally_id: str
    status: Literal['confirmed','rejected','unreviewed']
    winner: Literal['A','B'] | None = None
    serve_team: Literal['A','B'] | None = None

class VideoRecordsRequest(BaseModel):
    job_id: str
    match_id: str = Field(min_length=1, max_length=120)
    players: dict[str,str]
    labels: list[VerifiedLabel] = Field(max_length=10000)

@app.post('/api/video-rally-records')
def video_rally_records(payload: VideoRecordsRequest):
    if not payload.job_id.isalnum():
        raise HTTPException(400,'Invalid job ID')
    saved=DATA / (payload.job_id+'.json')
    if not saved.exists():
        raise HTTPException(404,'Analysis job not found or not finished')
    try:
        return build_records(json.loads(saved.read_text()), payload.match_id, payload.players,
                             [label.model_dump() for label in payload.labels])
    except ValueError as e:
        raise HTTPException(422,str(e))
