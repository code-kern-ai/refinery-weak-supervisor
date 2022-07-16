import weak_nlp
import pandas as pd


def get_cnlm_from_df(df: pd.DataFrame) -> weak_nlp.CNLM:
    vectors = []
    for source_id, df_sub_source in df.fillna("manual").groupby("source_id"):
        associations = []
        for _, row in df_sub_source.iterrows():
            associations.append(
                weak_nlp.ClassificationAssociation(
                    row.record_id, row.label_id, confidence=row.confidence
                )
            )
        vectors.append(
            weak_nlp.SourceVector(source_id, source_id == "manual", associations)
        )
    return weak_nlp.CNLM(vectors)


def get_enlm_from_df(df: pd.DataFrame) -> weak_nlp.ENLM:
    vectors = []
    for source_id, df_sub_source in df.fillna("manual").groupby("source_id"):
        associations = []
        for (
            record_id,
            label_id,
        ), df_sub_source_record_label in df_sub_source.groupby(
            ["record_id", "label_id"]
        ):
            chunk_start_idx = None
            chunk_end_idx = None
            for _, row in df_sub_source_record_label.iterrows():
                if row.is_beginning_token:
                    if chunk_start_idx is not None:
                        associations.append(
                            weak_nlp.ExtractionAssociation(
                                record_id,
                                label_id,
                                chunk_start_idx,
                                chunk_end_idx,
                                confidence=df_sub_source_record_label.confidence.iloc[
                                    0
                                ],
                            )
                        )
                    chunk_start_idx = row.token_index
                chunk_end_idx = row.token_index
            associations.append(
                weak_nlp.ExtractionAssociation(
                    record_id,
                    label_id,
                    chunk_start_idx,
                    chunk_end_idx,
                    confidence=df_sub_source_record_label.confidence.iloc[0],
                )
            )
        vectors.append(
            weak_nlp.SourceVector(source_id, source_id == "manual", associations)
        )
    return weak_nlp.ENLM(vectors)
