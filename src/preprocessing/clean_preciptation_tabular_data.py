import pandas as pd
from src.preprocessing.generate_time_sequence import generate_15min_sequence

def clean_preciptation_data(df: pd.DataFrame, fill_missing_dates=True) -> pd.DataFrame:
    """
    Transforms a precipitation wide-format DataFrame into a long-format DataFrame by unpivoting columns,
    cleaning the data, and restructuring it for easier analysis.

    The function performs the following steps:
    1. Drops initial unnecessary columns ('STATION', 'ELEMENT', 'LATITUDE', 'LONGITUDE', 'ELEVATION',
       'DlySum', 'DlySumMF', 'DlySumQF', 'DlySumS1', 'DlySumS2').
    2. Melts the DataFrame from wide to long format, using 'DATE' as the identifier variable and
       precipitation measurement columns as value variables.
    3. Combines the 'DATE' column and the time portion extracted from the 'Time_Attribute' column
       to create a full timestamp string.
    4. Extracts the measurement attribute (e.g., 'Val', 'MF', 'QF') from the 'Time_Attribute' column.
    5. Drops the original 'Time_Attribute' column.
    6. Pivots the DataFrame to wide format based on the extracted 'Attribute', using the combined
       timestamp as the index and 'Value' as the values.
    7. Removes the columns name attribute resulting from the pivot.
    8. Renames the pivoted columns for clarity ('DATE' to 'date', 'Val' to 'height', 'MF' to
       'measurement_flag', 'QF' to 'quality_flag').
    9. Drops unnecessary columns ('S1', 'S2') that resulted from the pivot.
    10. Converts the 'date' column to datetime objects.
    11. Replaces specific null indicators (-9999) in the 'height' column with pandas NA.
    12. Replaces any negative values in the 'height' column with pandas NA.
    13. Replaces specific quality flag values ('N') in the 'quality_flag' column with pandas NA.
    14. Filters the DataFrame to include only observations from January 1, 2014, onwards.
    15. Generates a complete sequence of 15-minute intervals covering the date range of the filtered data.
    16. Merges the data with the generated time sequence to explicitly include rows for missing time intervals.
    17. Cleans up columns after the merge (drops the old 'date' column and renames 'seq' to 'date').
    18. Converts the 'height' column to a numeric type, coercing errors.
    19. Converts 'height' values from hundredths to the standard unit (assuming inches/mm) by dividing by 100.
    20. Ensures the 'height' column has a float data type.
    
    -----
    Args:
        df (pd.DataFrame): The input DataFrame to be transformed, expected to be in a wide format
                           from precipitation data sources (like GHCN-D with time components).
        fill_missing_dates (bool): If True, fills in missing time intervals with NaN values for
    -----                           the 'height' column. Default is True.
    Returns:
        pd.DataFrame: The transformed long-format DataFrame with measurements separated by attribute,
                      cleaned, filtered, and with missing time intervals filled.
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
        'Val':'height',
        'MF':'measurement_flag',
        'QF':'quality_flag',
    }, axis='columns')

    # Drop columns
    drop_columns = ['S1','S2']
    df_final = df_final.drop(columns=drop_columns)

    # Parse datetime column
    df_final['date'] = pd.to_datetime(df_final['date'], format='%Y-%m-%d %H%M')

    # Replace null values
    df_final['height'] = df_final['height'].replace(-9999,pd.NA)

    # Replace any possible negative values with NA
    df_final['height'] = df_final['height'].where(df_final['height'] >= 0, pd.NA)

    # Replace failed to check negative precipitation with NA
    df_final['quality_flag'] = df_final['quality_flag'].replace('N', pd.NA)

    # Filter by date
    cutoff_date = pd.to_datetime('20140101')
    df_final = df_final[df_final['date']>=cutoff_date]

    if df_final.empty:
        raise ValueError("The DataFrame is empty after filtering by date. Please check the input data.")

    # Fill in missing dates
    if fill_missing_dates:
        # Generate a complete sequence of 15-minute intervals
        # covering the date range of the filtered data
        time_seq = generate_15min_sequence(df_final['date'].min(), df_final['date'].max())
        filled_seq_df = pd.DataFrame({'seq':time_seq}).merge(
            df_final,
            how='left',
            left_on='seq',
            right_on='date'
        )
        filled_seq_df = filled_seq_df.drop(columns=['date'])
        filled_seq_df = filled_seq_df.rename(columns={'seq':'date'})
        df_final = filled_seq_df.copy()

    # Convert height values to in and height dtype to float
    df_final['height'] = pd.to_numeric(df_final['height'], errors='coerce')
    df_final['height'] = df_final['height']/100
    df_final['height'] = df_final['height'].astype(float)

    return df_final