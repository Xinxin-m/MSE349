import pandas as pd
from typing import List, Tuple, Optional
def create_time_series_splits(
    df: pd.DataFrame,
    months_train: int = 6,
    months_test: int = 1,
    date_column: Optional[str] = None
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    # Make a copy so we don't modify the original order
    df = df.copy()

    # Determine which timestamp to use
    if date_column is not None:
        # Ensure the column is datetime
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

    while True:
        train_start = current_start
        train_end = train_start + pd.DateOffset(months=months_train)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=months_test)

        if test_end > last_date:
            break

        train_mask = (time_index >= train_start) & (time_index < train_end)
        test_mask = (time_index >= test_start) & (time_index < test_end)

        train_slice = df.loc[train_mask]
        test_slice = df.loc[test_mask]

        if not train_slice.empty and not test_slice.empty:
            train_dfs.append(train_slice)
            test_dfs.append(test_slice)

        # Slide start forward by one month
        current_start += pd.DateOffset(months=1)

    return train_dfs, test_dfs
