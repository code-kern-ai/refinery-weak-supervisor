from fastapi import FastAPI, HTTPException, responses, status
from pydantic import BaseModel

from controller import stats
from controller import integration
from submodules.model.business_objects import general

# API creation and description
app = FastAPI()


class WeakSupervisionRequest(BaseModel):
    project_id: str
    labeling_task_id: str
    user_id: str
    weak_supervision_task_id: str


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


@app.post("/fit_predict")
async def weakly_supervise(
    request: WeakSupervisionRequest,
) -> responses.PlainTextResponse:
    session_token = general.get_ctx_token()
    integration.fit_predict(
        request.project_id,
        request.labeling_task_id,
        request.user_id,
        request.weak_supervision_task_id,
    )
    general.remove_and_refresh_session(session_token)
    return responses.PlainTextResponse(status_code=status.HTTP_200_OK)


@app.post("/labeling_task_statistics")
async def calculate_task_stats(
    request: TaskStatsRequest,
) -> responses.PlainTextResponse:
    session_token = general.get_ctx_token()
    stats.calculate_quality_statistics_for_labeling_task(
        request.project_id, request.labeling_task_id, request.user_id
    )
    general.remove_and_refresh_session(session_token)
    return responses.PlainTextResponse(status_code=status.HTTP_200_OK)


@app.post("/source_statistics")
async def calculate_source_stats(
    request: SourceStatsRequest,
) -> responses.PlainTextResponse:
    session_token = general.get_ctx_token()
    has_coverage = stats.calculate_quantity_statistics_for_labeling_task_from_source(
        request.project_id, request.source_id, request.user_id
    )
    if has_coverage:
        stats.calculate_quality_statistics_for_source(
            request.project_id, request.source_id, request.user_id
        )
    general.remove_and_refresh_session(session_token)
    return responses.PlainTextResponse(status_code=status.HTTP_200_OK)


@app.post("/export_ws_stats")
async def export_ws_stats(request: ExportWsStatsRequest) -> responses.PlainTextResponse:
    session_token = general.get_ctx_token()
    status_code, message = integration.export_weak_supervision_stats(
        request.project_id, request.labeling_task_id
    )
    general.remove_and_refresh_session(session_token)

    if status_code != 200:
        raise HTTPException(status_code=status_code, detail=message)
    return responses.PlainTextResponse(status_code=status_code)


@app.get("/healthcheck")
def healthcheck() -> responses.PlainTextResponse:
    headers = {"APP": "OK"}
    database_test = general.test_database_connection()
    if not database_test.get("success"):
        headers["DATABASE"] = database_test.get("error")
    return responses.PlainTextResponse("OK", headers=headers)
