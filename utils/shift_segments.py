from dataclasses import dataclass

import pandas as pd

MS_PER_DAY = 86_400_000


@dataclass(frozen=True)
class TimeSegment:
    """Time segment in epoch milliseconds: start inclusive, end exclusive."""

    name: str
    start_ms: int | None
    end_ms: int | None


@dataclass(frozen=True)
class DayIdSplit:
    """Day-based split defined by UTC day_id = epoch_ms // MS_PER_DAY."""

    name: str
    train_day_ids: tuple[int, ...]
    val_day_ids: tuple[int, ...]
    test_day_ids: tuple[int, ...]
    split_day_id: int | None = None
    split_day_val_fraction: float = 0.0


def get_shift_aware_segments(dataset_name: str) -> list[TimeSegment]:
    """Return curated distribution-shift segments for known datasets."""
    if dataset_name == "NF-CSE-CIC-IDS2018-v3":
        # 2018-02-27 is a very low-support partial day, so it is excluded from
        # both coherent regimes. The post-shift regime starts at 2018-02-28.
        low_support_day_start = 1_519_689_600_000
        shift_start = 1_519_776_000_000
        return [
            TimeSegment(name="pre_shift", start_ms=None, end_ms=low_support_day_start),
            TimeSegment(name="post_shift", start_ms=shift_start, end_ms=None),
        ]
    return []


def get_shift_aware_day_split(
    dataset_name: str,
    segment: str,
) -> DayIdSplit | None:
    """Return a curated day split when a sequential split is degenerate."""
    if dataset_name == "NF-CSE-CIC-IDS2018-v3" and segment.strip() == "pre_shift":
        return DayIdSplit(
            name="day_split_v4",
            train_day_ids=(17576, 17577, 17578),
            val_day_ids=(17582,),
            test_day_ids=(17584, 17585),
            split_day_id=17583,
            split_day_val_fraction=0.5,
        )
    return None


def resolve_segment(dataset_name: str, segment: str) -> TimeSegment:
    segment = segment.strip()
    segments = get_shift_aware_segments(dataset_name)
    if not segments:
        raise ValueError(
            f"No shift-aware segments are defined for dataset={dataset_name!r}."
        )

    for candidate in segments:
        if candidate.name == segment:
            return candidate

    allowed = ", ".join(candidate.name for candidate in segments)
    raise ValueError(
        f"Unknown segment={segment!r} for dataset={dataset_name!r}. Allowed: {allowed}"
    )


def filter_df_by_segment(
    df: pd.DataFrame,
    *,
    timestamp_col: str,
    segment: TimeSegment,
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if segment.start_ms is not None:
        mask &= df[timestamp_col] >= segment.start_ms
    if segment.end_ms is not None:
        mask &= df[timestamp_col] < segment.end_ms
    return df.loc[mask].copy()
