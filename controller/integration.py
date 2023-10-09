import os
from typing import Any, Dict, List, Tuple, Optional, Union
import traceback
import pandas as pd
import pickle
from collections import defaultdict

from submodules.model.models import (
    LabelingTask,
    LabelingTaskLabel,
    RecordLabelAssociation,
    RecordLabelAssociationToken,
)
from . import util
from submodules.model import enums
from submodules.model.business_objects import (
    general,
    labeling_task,
    record_label_association,
    weak_supervision,
    labeling_task_label,
    information_source,
)

NO_LABEL_WS_PRECISION = 0.8


def __create_stats_lkp(
    project_id: str,
    labeling_task_id: str,
    overwrite_weak_supervision: Union[float, Dict[str, float]],
) -> Dict[Any, Any]:
    if isinstance(overwrite_weak_supervision, float):
        ws_weights = {}
        for heuristic_id in information_source.get_all_ids_by_labeling_task_id(
            project_id, labeling_task_id
        ):
            ws_weights[str(heuristic_id)] = overwrite_weak_supervision
    else:
        ws_weights = overwrite_weak_supervision

    ws_stats = {}
    for heuristic_id in ws_weights:
        label_ids = labeling_task_label.get_all_ids(project_id, labeling_task_id)
        for (label_id,) in label_ids:
            ws_stats[(heuristic_id, str(label_id))] = {
                "precision": ws_weights[heuristic_id]
            }
    return ws_stats


def fit_predict(
    project_id: str,
    labeling_task_id: str,
    user_id: str,
    weak_supervision_task_id: str,
    overwrite_weak_supervision: Optional[Dict[Any, Any]] = None,
):
    stats_lkp = None
    if overwrite_weak_supervision is not None:
        stats_lkp = __create_stats_lkp(
            project_id, labeling_task_id, overwrite_weak_supervision
        )
    elif not record_label_association.is_any_record_manually_labeled(
        project_id, labeling_task_id
    ):
        stats_lkp = __create_stats_lkp(
            project_id, labeling_task_id, NO_LABEL_WS_PRECISION
        )

    task_type, df = collect_data(project_id, labeling_task_id, True)
    try:
        if task_type == enums.LabelingTaskType.CLASSIFICATION.value:
            results = integrate_classification(df, stats_lkp)
        else:
            results = integrate_extraction(df, stats_lkp)
        weak_supervision.store_data(
            project_id,
            labeling_task_id,
            user_id,
            results,
            task_type,
            weak_supervision_task_id,
            with_commit=True,
        )
    except Exception:
        print(traceback.format_exc(), flush=True)
        general.rollback()
        weak_supervision.update_state(
            project_id,
            weak_supervision_task_id,
            enums.PayloadState.FAILED.value,
            with_commit=True,
        )


def export_weak_supervision_stats(
    project_id: str,
    labeling_task_id: str,
    overwrite_weak_supervision: Optional[Union[float, Dict[str, float]]] = None,
) -> Tuple[int, str]:
    if overwrite_weak_supervision is not None:
        ws_stats = __create_stats_lkp(
            project_id, labeling_task_id, overwrite_weak_supervision
        )
    elif not record_label_association.is_any_record_manually_labeled(
        project_id, labeling_task_id
    ):
        ws_stats = __create_stats_lkp(
            project_id, labeling_task_id, NO_LABEL_WS_PRECISION
        )
    else:
        task_type, df = collect_data(project_id, labeling_task_id, False)
        try:
            if task_type == enums.LabelingTaskType.CLASSIFICATION.value:
                cnlm = util.get_cnlm_from_df(df)
                stats_df = cnlm.quality_metrics()
            elif task_type == enums.LabelingTaskType.INFORMATION_EXTRACTION.value:
                enlm = util.get_enlm_from_df(df)
                stats_df = enlm.quality_metrics()
            else:
                return 404, f"Task type {task_type} not implemented"

            if len(stats_df) != 0:
                ws_stats = stats_df.set_index(["identifier", "label_name"]).to_dict(
                    orient="index"
                )
            else:
                return 404, "Can't compute weak supervision"

        except Exception:
            print(traceback.format_exc(), flush=True)
            general.rollback()
            return 500, "Internal server error"

    os.makedirs(os.path.join("/inference", project_id), exist_ok=True)
    with open(
        os.path.join(
            "/inference", project_id, f"weak-supervision-{labeling_task_id}.pkl"
        ),
        "wb",
    ) as f:
        pickle.dump(ws_stats, f)

    return 200, "OK"


def integrate_classification(df: pd.DataFrame, stats_lkp: Dict[Any, Any] = None):
    cnlm = util.get_cnlm_from_df(df)
    weak_supervision_results = cnlm.weakly_supervise(stats_lkp)
    return_values = defaultdict(list)
    for record_id, (
        label_id,
        confidence,
    ) in weak_supervision_results.dropna().items():
        return_values[record_id].append(
            {"label_id": label_id, "confidence": confidence}
        )
    return return_values


def integrate_extraction(df: pd.DataFrame, stats_lkp: Dict[Any, Any] = None):
    enlm = util.get_enlm_from_df(df)
    weak_supervision_results = enlm.weakly_supervise(stats_lkp)
    return_values = defaultdict(list)
    for record_id, preds in weak_supervision_results.items():
        for pred in preds:
            label, confidence, token_min, token_max = pred
            return_values[record_id].append(
                {
                    "label_id": label,
                    "confidence": confidence,
                    "token_index_start": token_min,
                    "token_index_end": token_max,
                }
            )
    return dict(return_values)


def collect_data(
    project_id: str, labeling_task_id: str, only_selected: bool
) -> Tuple[str, pd.DataFrame]:
    labeling_task_item = labeling_task.get(project_id, labeling_task_id)

    query_results = []
    if labeling_task_item.task_type == enums.LabelingTaskType.CLASSIFICATION.value:
        for information_source_item in labeling_task_item.information_sources:
            if only_selected and not information_source_item.is_selected:
                continue
            results = (
                record_label_association.get_all_classifications_for_information_source(
                    project_id, information_source_item.id
                )
            )
            query_results.extend(results)

        request_body = __jsonize_classification_associations(query_results)

        records_manual = record_label_association.get_manual_classifications_for_labeling_task_as_json(
            project_id, labeling_task_id
        )
        request_body.extend(records_manual)

    elif (
        labeling_task_item.task_type
        == enums.LabelingTaskType.INFORMATION_EXTRACTION.value
    ):
        for information_source_item in labeling_task_item.information_sources:
            if only_selected and not information_source_item.is_selected:
                continue
            results = record_label_association.get_all_extraction_tokens_for_information_source(
                project_id, information_source_item.id
            )
            query_results.extend(results)

        request_body = __jsonize_extraction_token_associations(query_results)
        records_manual = record_label_association.get_manual_extraction_tokens_for_labeling_task_as_json(
            project_id, labeling_task_id
        )
        request_body.extend(records_manual)
    return labeling_task_item.task_type, pd.DataFrame(request_body).drop_duplicates()


def __jsonize_extraction_token_associations(
    association_tuples: List[
        Tuple[
            RecordLabelAssociation,
            RecordLabelAssociationToken,
            LabelingTask,
            LabelingTaskLabel,
        ]
    ]
) -> List[Dict[str, Any]]:
    statistics_request_body = []
    for tuple in association_tuples:
        association, token, labeling_task, label = tuple
        source_id = None
        if association.source_id is not None:
            source_id = str(association.source_id)
        statistics_request_body.append(
            {
                "record_id": str(association.record_id),
                "source_id": source_id,
                "source_type": association.source_type,
                "confidence": association.confidence,
                "label_id": str(label.id),
                "token_index": token.token_index,
                "is_beginning_token": token.is_beginning_token,
            }
        )
    return statistics_request_body


def __jsonize_classification_associations(
    association_tuples: Tuple[
        RecordLabelAssociation,
        LabelingTask,
        LabelingTaskLabel,
    ]
) -> List[Dict[str, Any]]:
    statistics_request_body = []
    for tuple in association_tuples:
        association, labeling_task, label = tuple
        source_id = None
        if association.source_id is not None:
            source_id = str(association.source_id)
        statistics_request_body.append(
            {
                "record_id": str(association.record_id),
                "source_id": source_id,
                "source_type": association.source_type,
                "confidence": association.confidence,
                "label_id": str(label.id),
            }
        )
    return statistics_request_body
