# src/api/app.py
# FastAPI Application for Claim-Centric Multimodal News Verification.

import os
import time
import uuid
import asyncio
import tempfile
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

from src.api.schemas import (
    VerificationRequest,
    VerdictResponse,
    JobStatusResponse,
    HealthResponse,
    EvidenceItem,
    SourceCredibilityItem
)
from src.api.cache import ClaimCache
try:
    from main import NewsVerificationPipeline
except ImportError:
    NewsVerificationPipeline = None
from src.utils import setup_logger

logger = setup_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager that initializes and keeps AI models warm in memory.
    Runs once when the server boots, avoiding per-request cold starts.
    """
    logger.info("Initializing NewsVerificationPipeline for FastAPI server...")
    app.state.start_time = time.time()
    app.state.cache = ClaimCache(max_size=500, ttl_seconds=86400)
    app.state.jobs = {}  # In-memory background jobs: job_id -> dict

    # Allow custom config path via environment variable
    config_path = os.getenv('CONFIG_PATH', 'configs/model_config.yaml')
    if NewsVerificationPipeline is not None:
        try:
            app.state.pipeline = NewsVerificationPipeline(config_path=config_path)
            logger.info("Pipeline models successfully loaded into memory and ready.")
        except Exception as e:
            logger.error(f"Failed to initialize NewsVerificationPipeline: {e}")
            app.state.pipeline = None
    else:
        logger.warning("NewsVerificationPipeline not loaded (dependencies missing).")

    yield

    logger.info("Shutting down API server...")


app = FastAPI(
    title="Claim-Centric Multimodal News Verification API",
    description="Production REST API for verifying news claims using NLI, multi-source retrieval, and image forensics.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for web applications and browser extensions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_v1_router = APIRouter(prefix="/api/v1", tags=["Verification"])


def _format_verdict_response(raw_result: Dict[str, Any], cached: bool = False, execution_time: float = 0.0) -> VerdictResponse:
    """Format raw pipeline output into a clean Pydantic VerdictResponse with full UI compatibility."""
    final_v = raw_result.get('final_verdict', {})
    claim_obj = raw_result.get('claim')
    claim_text = claim_obj.original_text if hasattr(claim_obj, 'original_text') else str(claim_obj)

    top_evidence_items = []
    enriched_evidence_items = []
    nli_res = raw_result.get('nli_result', {})
    nli_details = (
        nli_res.get('evidence_details') or
        nli_res.get('individual_results') or
        []
    ) if isinstance(nli_res, dict) else []

    detail_map = {}
    for d in nli_details:
        if isinstance(d, dict):
            if 'evidence' in d and isinstance(d['evidence'], dict):
                ev_t = d['evidence'].get('text', '')
                detail_map[ev_t[:60]] = {
                    'verdict': d.get('prediction', 'neutral'),
                    'confidence': max(d.get('probabilities', {}).values()) if d.get('probabilities') else 0.5,
                    'probabilities': d.get('probabilities', {})
                }
            else:
                ev_t = d.get('text', '')
                detail_map[ev_t[:60]] = d

    for ev in raw_result.get('evidence', [])[:10]:
        t = ev.get('text', '')
        rel = 'neutral'
        rel_score = 0.5
        nli_scores = None
        matched = detail_map.get(t[:60])
        if matched:
            rel = matched.get('verdict', matched.get('relationship', 'neutral'))
            rel_score = matched.get('confidence', 0.5)
            nli_scores = matched.get('probabilities', matched.get('nli_scores'))

        breakdown = ev.get('scores_breakdown', {})
        credibility = breakdown.get('reliability_score', breakdown.get('reliability', ev.get('authority_score', 0.5)))

        source_tier = ev.get('source_tier')
        if not source_tier:
            is_wiki = bool(ev.get('url') and 'wikipedia.org' in (ev.get('url') or '').lower()) or ev.get('source') == 'wikipedia'
            if ev.get('is_official_primary'):
                source_tier = 'primary_origin'
            elif credibility >= 0.94:
                source_tier = 'fact_checker'
            elif credibility >= 0.91:
                source_tier = 'news_wire'
            elif is_wiki:
                source_tier = 'encyclopedic_record'
            elif credibility >= 0.85:
                source_tier = 'mainstream_press'
            elif credibility <= 0.50:
                source_tier = 'unverified_social'
            else:
                source_tier = 'general_web'

        item = {
            'source': ev.get('source', 'unknown'),
            'title': ev.get('title'),
            'text': t,
            'url': ev.get('url'),
            'score': ev.get('score'),
            'scores_breakdown': breakdown,
            'credibility_score': credibility,
            'stance': rel,
            'stance_score': rel_score,
            'relationship': rel,
            'nli_scores': nli_scores,
            'is_official_primary': ev.get('is_official_primary', False),
            'authority_score': ev.get('authority_score', credibility),
            'source_tier': source_tier
        }
        enriched_evidence_items.append(item)

        top_evidence_items.append(
            EvidenceItem(
                source=ev.get('source', 'unknown'),
                title=ev.get('title'),
                text=ev.get('text', ''),
                url=ev.get('url'),
                score=ev.get('score'),
                scores_breakdown=ev.get('scores_breakdown'),
                is_official_primary=ev.get('is_official_primary', False),
                authority_score=ev.get('authority_score', credibility),
                source_tier=source_tier
            )
        )

    # Format probabilities for frontend
    probs = nli_res.get('aggregated', {}).get('weighted_probabilities', {}) if isinstance(nli_res, dict) else {}
    probabilities = {
        'entailment': round(float(probs.get('entailment', 0.0)), 4),
        'contradiction': round(float(probs.get('contradiction', 0.0)), 4),
        'neutral': round(float(probs.get('neutral', 0.0)), 4),
    }

    # Format entities for frontend
    entities_dict = {}
    if hasattr(claim_obj, 'entities') and claim_obj.entities:
        try:
            entities_dict['ENTITIES'] = [str(e) for e in claim_obj.entities]
        except (TypeError, AttributeError):
            pass

    if hasattr(claim_obj, 'domain') and claim_obj.domain:
        try:
            domain_str = str(claim_obj.domain)
            if domain_str and domain_str != 'MagicMock':
                entities_dict['DOMAIN'] = [domain_str]
        except (TypeError, AttributeError):
            pass

    # Multi-Source Primary Origins resolution
    primary_origins_list = []
    if hasattr(claim_obj, 'primary_origins') and claim_obj.primary_origins:
        for po in claim_obj.primary_origins:
            primary_origins_list.append({
                'name': po.get('name', ''),
                'role': po.get('role', ''),
                'category': po.get('category', ''),
                'official_handles': po.get('official_handles', []),
                'official_domains': po.get('official_domains', [])
            })
    elif raw_result.get('primary_origins'):
        primary_origins_list = raw_result['primary_origins']
    else:
        try:
            from src.authority_sources import detect_authorities
            ents = claim_obj.entities if hasattr(claim_obj, 'entities') else []
            pos = detect_authorities(claim_text, entities=ents)
            for po in pos:
                primary_origins_list.append({
                    'name': po.get('name', ''),
                    'role': po.get('role', ''),
                    'category': po.get('category', ''),
                    'official_handles': po.get('official_handles', []),
                    'official_domains': po.get('official_domains', [])
                })
        except Exception:
            pass

    # Image scores & forensics
    img_res = raw_result.get('image_result')
    authenticity_score = None
    context_score = None
    ela_heatmap = None
    forensics_dict = None
    if img_res and isinstance(img_res, dict):
        auth_dict = img_res.get('authenticity', {})
        ctx_dict = img_res.get('context', {})
        if isinstance(auth_dict, dict):
            authenticity_score = auth_dict.get('authenticity_score')
            ela_details = auth_dict.get('ela_details', {})
            if isinstance(ela_details, dict):
                ela_heatmap = ela_details.get('heatmap_base64')
        if isinstance(ctx_dict, dict):
            context_score = ctx_dict.get('consistency_score', ctx_dict.get('overall_context_score'))

        forensics_dict = {
            'authenticity_score': authenticity_score,
            'context_score': context_score,
            'manipulation_type': auth_dict.get('manipulation_type', 'unknown') if isinstance(auth_dict, dict) else 'unknown',
            'ela': auth_dict.get('ela_details', {}) if isinstance(auth_dict, dict) else {},
            'noise': auth_dict.get('noise_consistency', {}) if isinstance(auth_dict, dict) else {},
            'exif': auth_dict.get('exif_metadata', {}) if isinstance(auth_dict, dict) else {},
            'spectral': auth_dict.get('spectral_details', {}) if isinstance(auth_dict, dict) else {},
            'context': ctx_dict if isinstance(ctx_dict, dict) else {}
        }

    def _safe_str(val):
        if val is None:
            return None
        try:
            s = str(val).strip()
        except (TypeError, AttributeError):
            return None
        # Filter out mock object string representations
        if not s or 'MagicMock' in s or 'Mock' in s:
            return None
        return s

    claimant_val = _safe_str(getattr(claim_obj, 'claimant', None))
    category_val = _safe_str(getattr(claim_obj, 'category', getattr(claim_obj, 'domain', None)))

    extracted_ocr = raw_result.get('extracted_ocr')
    if extracted_ocr is not None and not isinstance(extracted_ocr, dict):
        extracted_ocr = None

    final_verdict_str = final_v.get('final_verdict', 'Unverified')
    final_conf = round(float(final_v.get('final_confidence', 0.0)), 4)
    reasoning_str = final_v.get('reasoning', '')

    truth_analysis = final_v.get('truth_analysis')
    if not truth_analysis:
        try:
            from src.fusion_engine import FusionEngine
            engine = FusionEngine()
            truth_analysis = engine.generate_truth_analysis(
                claim=claim_text,
                final_verdict=final_verdict_str,
                final_confidence=final_conf,
                evidence=raw_result.get('evidence', []),
                primary_origins=primary_origins_list,
                nli_result=nli_res
            )
        except Exception as e:
            logger.warning(f"Truth analysis generation fallback: {e}")
            truth_analysis = None

    return VerdictResponse(
        claim=claim_text,
        final_verdict=final_verdict_str,
        final_confidence=final_conf,
        verdict=final_verdict_str,
        confidence=final_conf,
        text_verdict=final_v.get('text_verdict'),
        text_confidence=round(float(final_v.get('text_confidence', 0.0)), 4),
        image_verdict=final_v.get('image_verdict'),
        image_confidence=round(float(final_v.get('image_confidence', 0.0)), 4),
        evidence_count=len(raw_result.get('evidence', [])),
        top_evidence=top_evidence_items,
        evidence=enriched_evidence_items,
        probabilities=probabilities,
        entities=entities_dict,
        execution_time=round(execution_time, 3),
        authenticity_score=authenticity_score,
        context_score=context_score,
        extracted_ocr=extracted_ocr,
        detected_authorities=primary_origins_list,
        primary_origins=primary_origins_list,
        reasoning=reasoning_str,
        claimant=claimant_val,
        category=category_val,
        report=raw_result.get('report', ''),
        truth_analysis=truth_analysis,
        forensics=forensics_dict,
        ela_heatmap=ela_heatmap,
        cached=cached,
        timestamp=datetime.now().isoformat()
    )


async def _run_async_verification_job(
    job_id: str,
    input_data: str,
    input_type: str,
    temp_image_path: Optional[str] = None,
    max_evidence: Optional[int] = None,
    claim_text: Optional[str] = None
):
    """Background worker task that runs pipeline and updates job state."""
    app.state.jobs[job_id]['status'] = 'processing'
    app.state.jobs[job_id]['message'] = 'Running multi-source evidence retrieval and verification...'

    try:
        pipeline = app.state.pipeline
        t_start = time.time()
        raw_result = await asyncio.to_thread(
            pipeline.verify, input_data, input_type=input_type, max_evidence=max_evidence, claim_text=claim_text
        )
        exec_time = time.time() - t_start
        response_data = _format_verdict_response(raw_result, cached=False, execution_time=exec_time)

        # Store in cache
        if input_type in ('text', 'url'):
            app.state.cache.set(input_data, response_data.model_dump())

        app.state.jobs[job_id]['status'] = 'completed'
        app.state.jobs[job_id]['message'] = 'Verification completed successfully.'
        app.state.jobs[job_id]['result'] = response_data
        app.state.jobs[job_id]['completed_at'] = datetime.now().isoformat()

    except Exception as e:
        logger.error(f"Async verification job {job_id} failed: {e}")
        app.state.jobs[job_id]['status'] = 'failed'
        app.state.jobs[job_id]['message'] = f"Verification failed: {str(e)}"
        app.state.jobs[job_id]['completed_at'] = datetime.now().isoformat()
    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.unlink(temp_image_path)
            except Exception:
                pass


# Mount interactive web dashboard static assets
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/", tags=["Root"])
def root_info(request: Request):
    """Serve interactive web dashboard by default; return JSON API metadata when explicitly requested."""
    accept = request.headers.get("accept", "")
    format_param = request.query_params.get("format", "")

    # Return JSON API metadata only when explicitly requested via Accept header or query param
    if ("application/json" in accept and "text/html" not in accept) or format_param == "json":
        return {
            "service": "Claim-Centric Multimodal News Verification API",
            "version": "1.0.0",
            "status": "online",
            "documentation": "/docs",
            "web_dashboard": "/",
            "health_check": "/api/v1/health",
            "endpoints": {
                "verify": "POST /api/v1/verify",
                "verify_async": "POST /api/v1/verify/async",
                "job_status": "GET /api/v1/jobs/{job_id}",
                "sources": "GET /api/v1/sources"
            }
        }

    # Default to serving the interactive web dashboard
    if WEB_DIR.exists():
        return FileResponse(WEB_DIR / "index.html")

    return {
        "service": "Claim-Centric Multimodal News Verification API",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs"
    }



@app.get("/app", include_in_schema=False)
def serve_app():
    """Serve the interactive web verification dashboard."""
    if WEB_DIR.exists():
        return FileResponse(WEB_DIR / "index.html")
    raise HTTPException(status_code=404, detail="Web dashboard assets not found.")



@api_v1_router.get("/health", response_model=HealthResponse)
def health_check():
    """System health check, device status, and warm model states."""
    pipeline = getattr(app.state, 'pipeline', None)
    models_status = {
        "claim_extractor": pipeline is not None and hasattr(pipeline, 'claim_extractor'),
        "evidence_retriever": pipeline is not None and hasattr(pipeline, 'evidence_retriever'),
        "evidence_ranker": pipeline is not None and hasattr(pipeline, 'evidence_ranker'),
        "nli_verifier": pipeline is not None and hasattr(pipeline, 'nli_verifier'),
        "image_detector": pipeline is not None and hasattr(pipeline, 'image_detector'),
        "image_verifier": pipeline is not None and hasattr(pipeline, 'image_verifier'),
        "fusion_engine": pipeline is not None and hasattr(pipeline, 'fusion_engine'),
    }

    all_ready = all(models_status.values())
    uptime = time.time() - getattr(app.state, 'start_time', time.time())
    cache_entries = app.state.cache.size() if hasattr(app.state, 'cache') else 0

    return HealthResponse(
        status="healthy" if all_ready else "degraded",
        device="cuda" if (HAS_TORCH and torch and torch.cuda.is_available()) else "cpu",
        models_loaded=models_status,
        cache_entries=cache_entries,
        uptime_seconds=round(uptime, 2)
    )


@api_v1_router.get("/sources", response_model=List[SourceCredibilityItem])
def get_supported_sources():
    """List all supported evidence sources and their credibility ratings."""
    pipeline = getattr(app.state, 'pipeline', None)
    credibility_map = pipeline.evidence_ranker.source_credibility if (pipeline and hasattr(pipeline, 'evidence_ranker')) else {
        "reuters": 0.95,
        "ap_news": 0.95,
        "google_factcheck": 0.98,
        "snopes": 0.95,
        "politifact": 0.94,
        "factcheck": 0.94,
        "duckduckgo_factcheck": 0.92,
        "duckduckgo": 0.65,
        "web_search": 0.60
    }

    descriptions = {
        "google_factcheck": ("factchecktools.googleapis.com", "Fact-Checking API", "Official Google Fact Check Tools API indexing ClaimReview schemas from global fact-checking organizations."),
        "reuters": ("reuters.com", "Global News Wire", "Reuters Fact Check & News Service (authoritative primary source)."),
        "ap_news": ("apnews.com", "News Agency Wire", "Associated Press News Verification Desk."),
        "snopes": ("snopes.com", "Fact-Check Archive", "Snopes.com independent fact-checking archive."),
        "politifact": ("politifact.com", "Fact-Checking Registry", "PolitiFact Truth-O-Meter political statements database."),
        "factcheck": ("factcheck.org", "Consumer Fact-Check", "FactCheck.org non-partisan consumer advocate fact-check database."),
        "duckduckgo_factcheck": ("duckduckgo.com (Vetted)", "Journalism & Media", "Targeted live web search across vetted fact-checking and journalism domains."),
        "duckduckgo": ("duckduckgo.com", "General Search", "General DuckDuckGo live search results (fallback for real-time web discovery)."),
        "web_search": ("google.com", "Web Search", "Google Custom Search or general web index.")
    }

    items = []
    for src, score in credibility_map.items():
        if score <= 0.0 or src == 'wikipedia':
            continue
        info = descriptions.get(src, (src, "News / Reference", "Verified news and factual reference source."))
        items.append(
            SourceCredibilityItem(
                source=src,
                domain=info[0],
                category=info[1],
                credibility_score=score,
                description=info[2]
            )
        )
    return items


@api_v1_router.post("/verify", response_model=VerdictResponse)
async def verify_claim(
    request: VerificationRequest
):
    """
    Main verification endpoint for text claims and news URLs (JSON payload).
    """
    pipeline = getattr(app.state, 'pipeline', None)
    if not pipeline:
        raise HTTPException(status_code=503, detail="Verification pipeline is initializing.")

    claim_text = request.text.strip() if request.text else None
    news_url = request.url.strip() if request.url else None

    if not claim_text and not news_url:
        raise HTTPException(
            status_code=400,
            detail="At least one input required: 'text' claim or news 'url'."
        )

    input_type = 'url' if news_url else 'text'
    input_data = news_url if news_url else claim_text

    # Check cache (respect bypass_cache)
    if not request.async_mode and not request.bypass_cache:
        cached_data = app.state.cache.get(input_data)
        if cached_data:
            cached_data['cached'] = True
            if not cached_data.get('truth_analysis'):
                try:
                    from src.fusion_engine import FusionEngine
                    engine = FusionEngine()
                    cached_data['truth_analysis'] = engine.generate_truth_analysis(
                        claim=cached_data.get('claim', ''),
                        final_verdict=cached_data.get('final_verdict', cached_data.get('verdict', 'Unverified')),
                        final_confidence=float(cached_data.get('final_confidence', cached_data.get('confidence', 0.5))),
                        evidence=cached_data.get('evidence', cached_data.get('top_evidence', [])),
                        primary_origins=cached_data.get('primary_origins', [])
                    )
                except Exception:
                    pass
            return VerdictResponse(**cached_data)

    # If async requested, queue background job
    if request.async_mode:
        job_id = str(uuid.uuid4())
        app.state.jobs[job_id] = {
            'job_id': job_id,
            'status': 'queued',
            'message': 'Job queued for processing.',
            'result': None,
            'created_at': datetime.now().isoformat(),
            'completed_at': None
        }
        asyncio.create_task(_run_async_verification_job(job_id, input_data, input_type, max_evidence=request.max_evidence))
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "job_id": job_id,
                "status": "queued",
                "message": "Verification job submitted. Poll status at /api/v1/jobs/" + job_id
            }
        )

    # Synchronous execution in threadpool worker (does not block event loop)
    t_start = time.time()
    raw_result = await asyncio.to_thread(
        pipeline.verify, input_data, input_type=input_type, max_evidence=request.max_evidence
    )
    exec_time = time.time() - t_start
    response_data = _format_verdict_response(raw_result, cached=False, execution_time=exec_time)

    # Store in cache
    app.state.cache.set(input_data, response_data.model_dump())

    return response_data


@api_v1_router.post("/verify/image", response_model=VerdictResponse)
async def verify_image_claim(
    image: Optional[UploadFile] = File(None, description="Image file to analyze for manipulation and context"),
    file: Optional[UploadFile] = File(None, description="Alias for image file"),
    text: Optional[str] = Form(None, description="Optional associated claim text"),
    claim_text: Optional[str] = Form(None, description="Alias for associated claim text"),
    max_evidence: Optional[int] = Form(None, description="Max evidence sources to verify"),
    async_mode: bool = Form(False, description="If True, processes in the background")
):
    """
    Multimodal image verification endpoint (multipart/form-data).
    Performs ELA, spectral FFT, CNN forgery detection, and CLIP context alignment.
    """
    pipeline = getattr(app.state, 'pipeline', None)
    if not pipeline:
        raise HTTPException(status_code=503, detail="Verification pipeline is initializing.")

    img_file = image or file
    if not img_file:
        raise HTTPException(status_code=400, detail="Please upload an image file ('image' or 'file').")

    # Security validation: Verify file extension and MIME type
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.gif'}
    suffix = Path(img_file.filename).suffix.lower() if img_file.filename else '.jpg'
    content_type = (img_file.content_type or '').lower()

    if (suffix not in allowed_extensions) or (content_type and not content_type.startswith('image/')):
        raise HTTPException(
            status_code=400,
            detail="Image file required. Supported formats: JPG, PNG, WEBP, BMP, TIFF, GIF."
        )

    associated_text = text or claim_text
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        temp_image_path = tmp.name
        content = await img_file.read()
        tmp.write(content)

    input_data = temp_image_path
    input_type = 'image'

    if async_mode:
        job_id = str(uuid.uuid4())
        app.state.jobs[job_id] = {
            'job_id': job_id,
            'status': 'queued',
            'message': 'Image verification job queued.',
            'result': None,
            'created_at': datetime.now().isoformat(),
            'completed_at': None
        }
        asyncio.create_task(_run_async_verification_job(job_id, input_data, input_type, temp_image_path, max_evidence=max_evidence, claim_text=associated_text))
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "job_id": job_id,
                "status": "queued",
                "message": "Image verification job submitted. Poll status at /api/v1/jobs/" + job_id
            }
        )

    try:
        t_start = time.time()
        raw_result = await asyncio.to_thread(
            pipeline.verify, input_data, input_type=input_type, max_evidence=max_evidence, claim_text=associated_text
        )
        exec_time = time.time() - t_start
        response_data = _format_verdict_response(raw_result, cached=False, execution_time=exec_time)
        return response_data
    finally:
        if os.path.exists(temp_image_path):
            try:
                os.unlink(temp_image_path)
            except Exception:
                pass


@api_v1_router.post("/verify/async", response_model=JobStatusResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_async_verification(
    request: VerificationRequest
):
    """Convenience endpoint to submit an asynchronous verification job."""
    if not request.text and not request.url:
        raise HTTPException(status_code=400, detail="Provide 'text' or 'url' to verify.")

    input_data = request.text.strip() if request.text else request.url.strip()
    input_type = 'text' if request.text else 'url'

    job_id = str(uuid.uuid4())
    app.state.jobs[job_id] = {
        'job_id': job_id,
        'status': 'queued',
        'message': 'Job queued for processing.',
        'result': None,
        'created_at': datetime.now().isoformat(),
        'completed_at': None
    }

    asyncio.create_task(_run_async_verification_job(job_id, input_data, input_type))

    return JobStatusResponse(
        job_id=job_id,
        status='queued',
        message=f"Job queued. Check status at /api/v1/jobs/{job_id}",
        created_at=app.state.jobs[job_id]['created_at']
    )


@api_v1_router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    """Poll the status and retrieve results of an asynchronous verification job."""
    if job_id not in app.state.jobs:
        raise HTTPException(status_code=404, detail=f"Job ID '{job_id}' not found.")

    job = app.state.jobs[job_id]
    return JobStatusResponse(
        job_id=job['job_id'],
        status=job['status'],
        message=job.get('message'),
        result=job.get('result'),
        created_at=job['created_at'],
        completed_at=job.get('completed_at')
    )


@api_v1_router.delete("/cache")
def clear_rumor_cache():
    """Admin endpoint: Clear all cached claim verification results."""
    cleared = app.state.cache.clear()
    return {
        "status": "cleared",
        "cleared": cleared,
        "message": f"Cache cleared successfully. Removed {cleared} entries."
    }


app.include_router(api_v1_router)
