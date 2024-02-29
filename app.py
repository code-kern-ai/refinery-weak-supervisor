from fastapi import FastAPI, HTTPException, responses, status, Request
from pydantic import BaseModel
from typing import Union, Dict, Optional

import submodules.model.business_objects.general as general
from controller import stats
from controller import integration

# API creation and description
app = FastAPI()


class WeakSupervisionRequest(BaseModel):
    project_id: str
    labeling_task_id: str
    user_id: str
    weak_supervision_task_id: str
    overwrite_weak_supervision: Optional[Union[float, Dict[str, float]]]


class TaskStatsRequest(BaseModel):
    project_id: str
    labeling_task_id: str
    user_id: str


class SourceStatsRequest(BaseModel):
    project_id: str
    source_id: str
    user_id: str


class ExportWsStatsRequest(BaseModel):
    project_id: str
    labeling_task_id: str
    overwrite_weak_supervision: Optional[Union[float, Dict[str, float]]]


@app.middleware("http")
async def handle_db_session(request: Request, call_next):
    session_token = general.get_ctx_token()
    try:
        response = await call_next(request)
    finally:
        general.remove_and_refresh_session(session_token)

    return response


@app.post("/fit_predict")
def weakly_supervise(
    request: WeakSupervisionRequest,
) -> responses.PlainTextResponse:
    integration.fit_predict(
        request.project_id,
        request.labeling_task_id,
        request.user_id,
        request.weak_supervision_task_id,
        request.overwrite_weak_supervision,
    )
    return responses.PlainTextResponse(status_code=status.HTTP_200_OK)


@app.post("/labeling_task_statistics")
def calculate_task_stats(
    request: TaskStatsRequest,
) -> responses.PlainTextResponse:
    stats.calculate_quality_statistics_for_labeling_task(
        request.project_id, request.labeling_task_id, request.user_id
    )
    return responses.PlainTextResponse(status_code=status.HTTP_200_OK)


@app.post("/source_statistics")
def calculate_source_stats(
    request: SourceStatsRequest,
) -> responses.PlainTextResponse:
    has_coverage = stats.calculate_quantity_statistics_for_labeling_task_from_source(
        request.project_id, request.source_id, request.user_id
    )
    if has_coverage:
        stats.calculate_quality_statistics_for_source(
            request.project_id, request.source_id, request.user_id
        )
    return responses.PlainTextResponse(status_code=status.HTTP_200_OK)


@app.post("/export_ws_stats")
def export_ws_stats(request: ExportWsStatsRequest) -> responses.PlainTextResponse:
    status_code, message = integration.export_weak_supervision_stats(
        request.project_id, request.labeling_task_id, request.overwrite_weak_supervision
    )

    if status_code != 200:
        raise HTTPException(status_code=status_code, detail=message)
    return responses.PlainTextResponse(status_code=status_code)


@app.get("/healthcheck")
def healthcheck() -> responses.PlainTextResponse:
    text = ""
    status_code = status.HTTP_200_OK
    database_test = general.test_database_connection()
    if not database_test.get("success"):
        error_name = database_test.get("error")
        text += f"database_error:{error_name}:"
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    if not text:
        text = "OK"
    return responses.PlainTextResponse(text, status_code=status_code)
