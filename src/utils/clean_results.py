import pandas as pd

def clean_results_data(df: pd.DataFrame, imd_msoas: list[str]) -> pd.DataFrame:
    df_no_header = df[df['msoa21cd'] != 'msoa21cd']
    
    # Convert numeric columns safely without in-place mutation
    numeric_cols = [col for col in df_no_header.columns if col != 'msoa21cd']
    converted = df_no_header[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    # Combine back
    df_numeric = pd.concat([df_no_header[['msoa21cd']], converted], axis=1)
    
    # Drop NA and duplicates
    df_dedup = df_numeric.dropna().drop_duplicates(subset=['msoa21cd'])
    
    # Filter by IMD
    df_final = df_dedup[df_dedup['msoa21cd'].isin(imd_msoas)]
    
    return df_final

def clean_results() -> None:
    path = 'data/processed/national_bayesian_results.csv'
    df = pd.read_csv(path)
    
    imd_path = 'data/raw/imd/imd_2019_msoa.csv'
    imd_msoas = pd.read_csv(imd_path).iloc[:, 0].tolist()
    
    df_clean = clean_results_data(df, imd_msoas)
    
    print(f"Purified count: {len(df_clean)}")
    df_clean.to_csv(path, index=False)

if __name__ == "__main__":
    clean_results()
