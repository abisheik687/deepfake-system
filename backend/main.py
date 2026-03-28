"""
Internal trace:
- Wrong before: mixed legacy/live routes, duplicate health handlers, undefined logger imports, SPA serving in the API process, and on-demand model loading promises that were never implemented.
- Fixed now: single FastAPI runtime for upload analysis, lifespan-based model loading, consistent error envelopes, explicit CORS, and only the supported health/analyse routes.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from models.loader import model_lifespan
from routers.analyse import router as analyse_router
from routers.health import router as health_router
from utils.file_utils import AppError
from utils.logger import get_logger


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with model_lifespan(app):
        logger.info('application_started', extra={'app_name': settings.app_name})
        yield
        logger.info('application_stopped', extra={'app_name': settings.app_name})


app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=['GET', 'POST', 'OPTIONS'],
    allow_headers=['*'],
)

app.include_router(health_router)
app.include_router(analyse_router)


@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={'error': exc.message, 'code': exc.code})


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    first = exc.errors()[0] if exc.errors() else {}
    location = '.'.join(str(part) for part in first.get('loc', []) if part != 'body')
    message = first.get('msg', 'Invalid request')
    if location:
        message = f'{location}: {message}'
    return JSONResponse(status_code=422, content={'error': message, 'code': 'INVALID_REQUEST'})


@app.exception_handler(Exception)
async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception('unhandled_exception', extra={'error': str(exc)})
    return JSONResponse(
        status_code=500,
        content={'error': 'Analysis failed - try again', 'code': 'INTERNAL_ERROR'},
    )


@app.get('/')
async def root() -> dict[str, str]:
    return {'name': settings.app_name, 'version': settings.app_version, 'docs': '/docs'}


if __name__ == '__main__':
    import uvicorn

    uvicorn.run('main:app', host=settings.host, port=settings.port, reload=True)
