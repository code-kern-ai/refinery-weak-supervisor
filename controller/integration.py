import os
from typing import Any, Dict, List, Tuple
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
)


def fit_predict(
    project_id: str, labeling_task_id: str, user_id: str, weak_supervision_task_id: str
):
    task_type, df = collect_data(project_id, labeling_task_id, True)
    try:
        if task_type == enums.LabelingTaskType.CLASSIFICATION.value:
            results = integrate_classification(df)

        else:
            results = integrate_extraction(df)
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
    project_id: str, labeling_task_id: str
) -> Tuple[int, str]:

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
            stats_lkp = stats_df.set_index(["identifier", "label_name"]).to_dict(
                orient="index"
            )
        else:
            return 404, "Can't compute weak supervision"

        os.makedirs(os.path.join("/inference", project_id), exist_ok=True)
        with open(
            os.path.join(
                "/inference", project_id, f"weak-supervision-{labeling_task_id}.pkl"
            ),
            "wb",
        ) as f:
            pickle.dump(stats_lkp, f)

    except Exception:
        print(traceback.format_exc(), flush=True)
        general.rollback()
        return 500, "Internal server error"
    return 200, "OK"


def integrate_classification(df: pd.DataFrame):
    cnlm = util.get_cnlm_from_df(df)
    weak_supervision_results = cnlm.weakly_supervise()
    return_values = defaultdict(list)
    for record_id, (
        label_id,
        confidence,
    ) in weak_supervision_results.dropna().items():
        return_values[record_id].append(
            {"label_id": label_id, "confidence": confidence}
        )
    return return_values


def integrate_extraction(df: pd.DataFrame):
    enlm = util.get_enlm_from_df(df)
    weak_supervision_results = enlm.weakly_supervise()
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
        for information_source in labeling_task_item.information_sources:
            if only_selected and not information_source.is_selected:
                continue
            results = (
                record_label_association.get_all_classifications_for_information_source(
                    project_id, information_source.id
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
        for information_source in labeling_task_item.information_sources:
            if only_selected and not information_source.is_selected:
                continue
            results = record_label_association.get_all_extraction_tokens_for_information_source(
                project_id, information_source.id
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
