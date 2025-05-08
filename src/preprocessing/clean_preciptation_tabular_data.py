import pandas as pd

def unpivot_preciptation_v2_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms a v2 precipitation wide-format DataFrame into a long-format DataFrame by unpivoting columns, 
    cleaning the data, and restructuring it for easier analysis.

    The function performs the following steps:
    1. Drops unnecessary columns.
    2. Melts the DataFrame to long format based on the 'DATE' column.
    3. Combines 'DATE' and 'Time_Attribute' for a new timestamp.
    4. Extracts 'Attribute' from the 'Time_Attribute' column.
    5. Pivots the data to separate out different measurements.
    6. Renames the columns for clarity.
    7. Replace -9999 (null values) by NA
    8. Filter out observation prior to year 2014

    Args:
    -----
    df (pd.DataFrame): The input DataFrame to be unpivoted.

    Returns:
    --------
    pd.DataFrame: The transformed long-format DataFrame with measurements separated by attribute.
    """

    columns_to_trop = ['STATION', 'ELEMENT','LATITUDE', 'LONGITUDE', 'ELEVATION', 'DlySum', 'DlySumMF', 'DlySumQF', 'DlySumS1', 'DlySumS2']
    df = df.drop(columns=columns_to_trop)

    # Identifying columns to melt
    id_vars = ['DATE']
    value_vars = [col for col in df.columns if col not in id_vars]

    # Melting to long format
    df_long = df.melt(id_vars=id_vars,
                                        value_vars=value_vars,
                                        var_name='Time_Attribute',
                                        value_name='Value')

    # Add Time to DATE column and extract Attribute from 'Time_Attribute'
    df_long['DATE'] = df_long['DATE'].astype(str) + ' ' + df_long['Time_Attribute'].str[:4]
    df_long['Attribute'] = df_long['Time_Attribute'].str[4:]

    df_long = df_long.drop(columns=['Time_Attribute'])

    # Pivoting to separate Value, Measurement Flag, Quality Flag, and Source Flags
    df_final = df_long.pivot_table(
        index=['DATE'], 
        columns='Attribute', 
        values='Value', 
        aggfunc='first'
    ).reset_index()

    df_final.columns.name = None 
    df_final = df_final.rename({
        'DATE':'date',
        'Val':'height'
    }, axis='columns')

    # Drop columns
    keep_columns = ['date','height']
    df_final = df_final[keep_columns]

    # Parse datetime column
    df_final['date'] = pd.to_datetime(df_final['date'], format='%Y-%m-%d %H%M')

    # Replace null values
    df_final['height'] = df_final['height'].replace(-9999,pd.NA)

    # Filter by date
    cutoff_date = pd.to_datetime('20140101')
    df_final = df_final[df_final['date']>=cutoff_date]
    
    return df_final