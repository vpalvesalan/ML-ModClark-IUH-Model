import pandas as pd

def glimpse(df: pd.DataFrame) -> None:
    """
    Displays a summary of a pandas DataFrame, similar to pd.DataFrame.info().
    It shows the number of rows, columns, data types, null counts, and the first 
    5 values of each column.

    Parameters:
    -----------
    df : pd.DataFrame
        The pandas DataFrame to be inspected.

    Returns:
    --------
    None
    """
    # Print the number of rows and columns
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    # Determine the maximum column name and dtype lengths for formatting
    dtypes = df.dtypes.astype(str).tolist()
    dtypes_len = max(len(dtype) for dtype in dtypes) + 1

    columns = df.columns.astype(str).tolist()
    columns_len = max(len(col) for col in columns) + 1
    columns_len = min(columns_len, 29)  # Limit the column name length to 29

    # Build the header for column details
    col_info = [' ' * columns_len + ' Null Count  Dtype' + ' ' * (dtypes_len - 4) + 'First Values',
                ' ' * columns_len + ' ----------  -----' + ' ' * (dtypes_len - 4) + '-------------']

    # Add details for each column
    for col in df.columns:
        col_edited = col[:27] + '...' if len(col) > 29 else col  # Shorten long column names
        null_count = df[col].isnull().sum()
        dtype = df[col].dtype
        first_values = ', '.join(map(str, df[col].head().values))  # Format first 5 values

        # Append the formatted column information
        col_info.append(f"{col_edited:<{columns_len}} {null_count:<10}  {str(dtype):<{dtypes_len}} [{first_values}]")

    # Print the column details
    for col in col_info:
        print(col)
