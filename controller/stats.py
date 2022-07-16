import os
from typing import Dict, Optional
import pandas as pd
import requests
from submodules.model.business_objects.organization import get_organization_id
from . import util
from controller import integration
from submodules.model import enums
from submodules.model.business_objects import (
    information_source,
    labeling_task,
    notification,
    project,
)
import weak_nlp

WEBSOCKET_ENDPOINT = os.getenv("WS_NOTIFY_ENDPOINT")


def send_organization_update(
    project_id: str,
    message: str,
    is_global: bool = False,
    organization_id: Optional[str] = None,
) -> None:

    if not WEBSOCKET_ENDPOINT:
        print(
            "- WS_NOTIFY_ENDPOINT not set -- did you run the start script?", flush=True
        )
        return

    if is_global:
        message = f"GLOBAL:{message}"
    else:
        message = f"{project_id}:{message}"
    if not organization_id:
        project_item = project.get(project_id)
        organization_id = str(project_item.organization_id)

    req = requests.post(
        f"{WEBSOCKET_ENDPOINT}/notify",
        json={
            "organization": organization_id,
            "message": message,
        },
    )
    if req.status_code != 200:
        print("Could not send notification update", flush=True)


def send_warning_no_reference_data(project_id: str, user_id: str):
    notification.create(
        project_id,
        user_id,
        "You have no labeled data. Can't compute true positive-related statistics.",
        "WARNING",
        enums.NotificationType.MISSING_REFERENCE_DATA.value,
        with_commit=True,
    )
    organization_id = get_organization_id(project_id, user_id)
    if organization_id:
        send_organization_update(
            project_id, f"notification_created:{user_id}", True, organization_id
        )


def send_warning_no_coverage_data(project_id: str, user_id: str):
    notification.create(
        project_id,
        user_id,
        "Your heuristics hits no records in the project. Can't compute statistics.",
        "WARNING",
        enums.NotificationType.MISSING_REFERENCE_DATA.value,
        with_commit=True,
    )
    organization_id = get_organization_id(project_id, user_id)
    if organization_id:
        send_organization_update(
            project_id, f"notification_created:{user_id}", True, organization_id
        )


def calculate_quality_statistics_for_labeling_task(
    project_id: str, task_id: str, user_id: str
):
    labeling_task_item = labeling_task.get_labeling_task_by_id_only(task_id)
    _, df = integration.collect_data(
        labeling_task_item.project_id, labeling_task_item.id, False
    )
    exclusion_ids = information_source.get_exclusion_record_ids_for_task(task_id)
    df = df.loc[~df["record_id"].isin(exclusion_ids)]
    try:
        if labeling_task_item.task_type == enums.LabelingTaskType.CLASSIFICATION.value:
            statistics = classification_quality(df)
        else:
            statistics = extraction_quality(df)
        for source_id, statistics_item in statistics.items():
            information_source.update_quality_stats(
                labeling_task_item.project_id,
                source_id,
                statistics_item,
                with_commit=True,
            )
    except weak_nlp.shared.exceptions.MissingReferenceException:
        send_warning_no_reference_data(project_id, user_id)


def calculate_quality_statistics_for_source(
    project_id: str, source_id: str, user_id: str
):
    labeling_task_item = labeling_task.get_labeling_task_by_source_id(source_id)
    _, df = integration.collect_data(
        labeling_task_item.project_id, labeling_task_item.id, False
    )
    exclusion_ids = information_source.get_exclusion_record_ids(source_id)
    df = df.loc[~df["record_id"].isin(exclusion_ids)]
    try:
        if labeling_task_item.task_type == enums.LabelingTaskType.CLASSIFICATION.value:
            statistics = classification_quality(df)
        else:
            statistics = extraction_quality(df)
        stats = statistics.get(source_id)
        if stats is not None:
            information_source.update_quality_stats(
                labeling_task_item.project_id, source_id, stats, with_commit=True
            )
    except weak_nlp.shared.exceptions.MissingReferenceException:
        send_warning_no_reference_data(project_id, user_id)


def calculate_quantity_statistics_for_labeling_task_from_source(
    project_id: str, source_id: str, user_id: str
) -> bool:
    labeling_task_item = labeling_task.get_labeling_task_by_source_id(source_id)
    _, df = integration.collect_data(
        labeling_task_item.project_id, labeling_task_item.id, False
    )
    if labeling_task_item.task_type == enums.LabelingTaskType.CLASSIFICATION.value:
        statistics = classification_quantity(df)
    else:
        statistics = extraction_quantity(df)

    if source_id not in statistics:
        send_warning_no_coverage_data(project_id, user_id)
        return False
    information_source.delete_stats(labeling_task_item.project_id, source_id)
    for source_id, statistics_item in statistics.items():
        information_source.update_quantity_stats(
            labeling_task_item.project_id, source_id, statistics_item, with_commit=True
        )
    return True


def classification_quantity(df: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, int]]]:
    cnlm = util.get_cnlm_from_df(df)
    quantity_df = cnlm.quantity_metrics()

    stats = {}
    if len(quantity_df) > 0:
        for source_id, quality_df_sub_source in quantity_df.groupby("identifier"):
            stats[source_id] = {}
            for label_id, quality_df_sub_source_label in quality_df_sub_source.groupby(
                "label_name"
            ):
                row = quality_df_sub_source_label.iloc[0]
                stats[source_id][label_id] = {
                    "record_coverage": int(row["record_coverage"]),
                    "source_conflicts": int(row["source_conflicts"]),
                    "source_overlaps": int(row["source_overlaps"]),
                    "total_hits": int(row["record_coverage"]),
                }
    return stats


def extraction_quantity(df: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, int]]]:
    enlm = util.get_enlm_from_df(df)
    quantity_df = enlm.quantity_metrics()

    stats = {}
    if len(quantity_df) > 0:
        for source_id, quality_df_sub_source in quantity_df.groupby("identifier"):
            stats[source_id] = {}
            for label_id, quality_df_sub_source_label in quality_df_sub_source.groupby(
                "label_name"
            ):
                row = quality_df_sub_source_label.iloc[0]
                stats[source_id][label_id] = {
                    "record_coverage": int(row["record_coverage"]),
                    "source_conflicts": int(row["source_conflicts"]),
                    "source_overlaps": int(row["source_overlaps"]),
                    "total_hits": int(row["total_hits"]),
                }
    return stats


def classification_quality(df: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, int]]]:
    cnlm = util.get_cnlm_from_df(df)
    quality_df = cnlm.quality_metrics()
    stats = {}
    if len(quality_df) > 0:
        for source_id, quality_df_sub_source in quality_df.groupby("identifier"):
            stats[source_id] = {}
            for label_id, quality_df_sub_source_label in quality_df_sub_source.groupby(
                "label_name"
            ):
                row = quality_df_sub_source_label.iloc[0]
                stats[source_id][label_id] = {
                    "true_positives": int(row["true_positives"]),
                    "false_positives": int(row["false_positives"]),
                    "false_negatives": 0,
                }
    return stats


def extraction_quality(df: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, int]]]:
    enlm = util.get_enlm_from_df(df)
    quality_df = enlm.quality_metrics()
    stats = {}
    if len(quality_df) > 0:
        for source_id, quality_df_sub_source in quality_df.groupby("identifier"):
            stats[source_id] = {}
            for label_id, quality_df_sub_source_label in quality_df_sub_source.groupby(
                "label_name"
            ):
                row = quality_df_sub_source_label.iloc[0]
                stats[source_id][label_id] = {
                    "true_positives": int(row["true_positives"]),
                    "false_positives": int(row["false_positives"]),
                    "false_negatives": int(row["false_negatives"]),
                }
    return stats
