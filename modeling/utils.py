import pandas as pd
from typing import List, Tuple, Optional

def create_time_series_splits(
    df: pd.DataFrame,
    months_train: float = 6,
    months_test: float = 1,
    date_column: Optional[str] = None
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    df = df.copy()

    # pick the timestamp column / index
    if date_column is not None:
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(by=date_column)
        time_index = df[date_column]
    else:
        df = df.sort_index()
        time_index = df.index

    train_dfs = []
    test_dfs = []

    first_date = time_index.min()
    last_date = time_index.max()
    current_start = first_date

    # helper to turn “months” (int or float) → offset
    def to_offset(m: float):
        if float(m).is_integer():
            # exact integer → calendar‐month offset
            return pd.DateOffset(months=int(m))
        else:
            # fractional → convert to days (30 days per month)
            days = float(m) * 30
            return pd.Timedelta(days=days)

    train_offset = to_offset(months_train)
    test_offset  = to_offset(months_test)
    slide_offset = test_offset  # slide by exactly months_test each loop

    while True:
        train_start = current_start
        train_end   = train_start + train_offset

        test_start = train_end
        test_end   = test_start + test_offset

        # stop when next test window would exceed data range
        if test_end > last_date:
            break

        train_mask = (time_index >= train_start) & (time_index < train_end)
        test_mask  = (time_index >= test_start)  & (time_index < test_end)

        train_slice = df.loc[train_mask]
        test_slice  = df.loc[test_mask]

        if not train_slice.empty and not test_slice.empty:
            train_dfs.append(train_slice)
            test_dfs.append(test_slice)

        # slide by months_test (e.g. 0.5 → 15 days, 1 → one calendar month)
        current_start = current_start + slide_offset

    return train_dfs, test_dfs
