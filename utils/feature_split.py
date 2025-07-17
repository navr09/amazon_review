import pandas as pd

def feature_type_split(df_features,total_features):

            numerical_features = []
            categorical_features = [] 
            # Check if dtype is numerical
            for col in total_features:
                if col not in df_features.columns:
                    print(f"Warning: Column '{col}' not found in DataFrame")
                    continue
                    
                # Check for numerical columns (int, float, bool)
                if pd.api.types.is_numeric_dtype(df_features[col]):
                    numerical_features.append(col)
                # Check for categorical (strings, objects, or categorical dtypes)
                elif pd.api.types.is_string_dtype(df_features[col]) or pd.api.types.is_categorical_dtype(df_features[col]):
                    categorical_features.append(col)
                else:
                    print(f"Warning: Column '{col}' has unhandled dtype: {df_features[col].dtype}")
            return numerical_features,categorical_features