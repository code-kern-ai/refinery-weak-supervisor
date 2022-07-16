from fastapi import FastAPI

from controller import stats
from controller import integration
from submodules.model.business_objects import general

# API creation and description
app = FastAPI()

from pydantic import BaseModel


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


@app.post("/fit_predict")
async def weakly_supervise(request: WeakSupervisionRequest) -> int:
    session_token = general.get_ctx_token()
    integration.fit_predict(
        request.project_id,
        request.labeling_task_id,
        request.user_id,
        request.weak_supervision_task_id,
    )
    general.remove_and_refresh_session(session_token)
    return None, 200


@app.post("/labeling_task_statistics")
async def calculate_stats(request: TaskStatsRequest):
    session_token = general.get_ctx_token()
    stats.calculate_quality_statistics_for_labeling_task(
        request.project_id, request.labeling_task_id, request.user_id
    )
    general.remove_and_refresh_session(session_token)
    return None, 200


@app.post("/source_statistics")
async def calculate_stats(request: SourceStatsRequest):
    session_token = general.get_ctx_token()
    has_coverage = stats.calculate_quantity_statistics_for_labeling_task_from_source(
        request.project_id, request.source_id, request.user_id
    )
    if has_coverage:
        stats.calculate_quality_statistics_for_source(
            request.project_id, request.source_id, request.user_id
        )
    general.remove_and_refresh_session(session_token)
    return None, 200
