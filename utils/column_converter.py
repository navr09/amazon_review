import pandas as pd

def convert_columns(df, conversion_rules):
    """
    Converts specified columns in a DataFrame based on given conversion rules.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame.
    conversion_rules : list of tuples
        Each tuple should be (list_of_columns, target_type).
        Example: [(['price', 'date'], 'numeric'), (['date'], 'datetime')]

    Returns:
    --------
    pd.DataFrame
        DataFrame with converted columns.
    """
    df_converted = df.copy()
    
    for columns, target_type in conversion_rules:
        for col in columns:
            if col not in df.columns:
                print(f"Warning: Column '{col}' not found. Skipping.")
                continue
            
            original_dtype = df_converted[col].dtype
            
            try:
                if target_type == 'numeric':
                    df_converted[col] = pd.to_numeric(df_converted[col], errors='raise')
                elif target_type == 'datetime':
                    df_converted[col] = pd.to_datetime(df_converted[col], errors='raise')
                elif target_type == 'object':
                    df_converted[col] = df_converted[col].astype('string')
                else:
                    print(f"Warning: Unsupported target type '{target_type}'. Skipping column '{col}'.")
                    continue
                
                print(f"Converted '{col}' from {original_dtype} to {target_type}.")
            except Exception as e:
                print(f"Error converting '{col}' to {target_type}: {str(e)}. Keeping original dtype.")
    
    return df_converted