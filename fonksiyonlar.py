#--------------------
"""
EPIAS Energy Market Data Processing Module

This module processes Turkish energy market data from multiple sources:
- PTF (Market Clearing Price)
- Load Forecast Plan
- KGUP (Finalized Daily Generation Plan)
- USD/TRY exchange rates
- Natural gas prices
"""

import pandas as pd
import yfinance as yf
from typing import List, Tuple
from datetime import timedelta


def read_data_files(
        ptf_path: str,
        yuk_path: str,
        kgup_paths: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read all input data files.

    Parameters:
    -----------
    ptf_path : str
        Path to PTF (Market Clearing Price) CSV file
    yuk_path : str
        Path to Load Forecast Plan CSV file
    kgup_paths : List[str]
        List of paths to KGUP (Finalized Daily Generation Plan) CSV files

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        PTF dataframe, Load dataframe, KGUP dataframe
    """
    # Read PTF and Load data
    ptf_df = pd.read_csv(ptf_path, sep=";")
    yuk_df = pd.read_csv(yuk_path, sep=";")

    # Read and concatenate KGUP files
    kgup_dfs = []
    for file in kgup_paths:
        df = pd.read_csv(file, sep=";")
        kgup_dfs.append(df)

    kgup_df = pd.concat(kgup_dfs, ignore_index=True)
    kgup_df = kgup_df.drop_duplicates(subset=['Tarih', 'Saat']).reset_index(drop=True)

    return ptf_df, yuk_df, kgup_df


def merge_dataframes(
        ptf_df: pd.DataFrame,
        yuk_df: pd.DataFrame,
        kgup_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge all dataframes on Date and Hour columns.

    Parameters:
    -----------
    ptf_df : pd.DataFrame
        PTF dataframe
    yuk_df : pd.DataFrame
        Load forecast dataframe
    kgup_df : pd.DataFrame
        KGUP dataframe

    Returns:
    --------
    pd.DataFrame
        Merged dataframe with all data sources
    """
    # First merge: PTF and Load
    merge_1 = pd.merge(ptf_df, yuk_df, on=["Tarih", "Saat"], how="inner")

    # Standardize date format
    merge_1['Tarih'] = pd.to_datetime(merge_1['Tarih'], dayfirst=True, errors='coerce').dt.normalize()
    kgup_df['Tarih'] = pd.to_datetime(kgup_df['Tarih'], dayfirst=True, errors='coerce').dt.normalize()

    # Standardize hour format
    merge_1['Saat'] = merge_1['Saat'].astype(str).str.strip().str[:5]
    kgup_df['Saat'] = kgup_df['Saat'].astype(str).str.strip().str[:5]

    # Second merge: Add KGUP data
    df_final = pd.merge(merge_1, kgup_df, on=["Tarih", "Saat"], how="inner").reset_index(drop=True)
    df_final = df_final.sort_values(by=["Tarih", "Saat"]).reset_index(drop=True)

    return df_final


def clean_currency(x) -> float:
    """
    Convert Turkish number format to float.
    Handles dot (.) as thousands separator and comma (,) as decimal separator.

    Parameters:
    -----------
    x : str or numeric
        Value to convert

    Returns:
    --------
    float
        Converted numeric value
    """
    if isinstance(x, str):
        # Remove thousands separator (dots)
        x = x.replace('.', '')
        # Convert decimal separator (comma) to dot
        x = x.replace(',', '.')
    return float(x)


def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert string columns to numeric values.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    pd.DataFrame
        Dataframe with converted numeric columns
    """
    # Get all columns except Tarih and Saat
    cols_to_convert = [col for col in df.columns if col not in ['Tarih', 'Saat']]

    for col in cols_to_convert:
        df[col] = df[col].apply(clean_currency)

    return df


def add_usd_exchange_rate(
        df: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
) -> pd.DataFrame:
    """
    Add USD/TRY exchange rate data from Yahoo Finance.

    Parameters:
    -----------
    df : pd.DataFrame
        Main dataframe
    start_date : pd.Timestamp
        Start date for exchange rate data
    end_date : pd.Timestamp
        End date for exchange rate data

    Returns:
    --------
    pd.DataFrame
        Dataframe with USD/TRY exchange rate added
    """
    # Download USD/TRY data
    usd_data = yf.download('TRY=X', start=start_date, end=end_date + timedelta(days=5))
    usd_data = usd_data['Close'].reset_index()
    usd_data.columns = ['Tarih', 'Dolar_Kuru']

    # Standardize date format
    usd_data['Tarih'] = pd.to_datetime(usd_data['Tarih']).dt.normalize()
    usd_data['Tarih'] = usd_data['Tarih'].dt.tz_localize(None)

    # Fill missing dates
    all_dates = pd.DataFrame({
        'Tarih': pd.date_range(start=start_date, end=end_date, freq='D')
    })
    all_dates['Tarih'] = all_dates['Tarih'].dt.normalize().dt.tz_localize(None)

    usd_data = pd.merge(all_dates, usd_data, on='Tarih', how='left')
    usd_data['Dolar_Kuru'] = usd_data['Dolar_Kuru'].ffill().bfill()

    # Merge with main dataframe
    df = pd.merge(df, usd_data, on='Tarih', how='left')

    return df


def add_natural_gas_prices(
        df: pd.DataFrame,
        threshold_date: str = '2025-07-01',
        price_before: float = 1127.82,
        price_after: float = 1409.77
) -> pd.DataFrame:
    """
    Add natural gas prices based on date threshold.

    Parameters:
    -----------
    df : pd.DataFrame
        Main dataframe
    threshold_date : str
        Date threshold for price change
    price_before : float
        Price for dates before or equal to threshold
    price_after : float
        Price for dates after threshold

    Returns:
    --------
    pd.DataFrame
        Dataframe with natural gas prices added
    """
    threshold = pd.Timestamp(threshold_date)

    df['dogalgaz_fiyatlari_Mwh'] = [
        price_before if tarih <= threshold else price_after
        for tarih in df['Tarih']
    ]

    return df


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove unnecessary columns from the dataset.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    pd.DataFrame
        Dataframe with unnecessary columns removed
    """
    drop_list = [
        'PTF (USD/MWh)', 'PTF (EUR/MWh)',
        'Toplam(MWh)',
        'Nafta', 'Fueloil',
        'Taş Kömür', 'Diğer'
    ]

    existing_drop = [col for col in drop_list if col in df.columns]

    if existing_drop:
        df = df.drop(columns=existing_drop)

    return df


def process_epias_data(
        ptf_path: str = "data_s/Piyasa_Takas_Fiyati(PTF).csv",
        yuk_path: str = "data_s/Yuk_Tahmin_Plani.csv",
        kgup_paths: List[str] = None,
        output_path: str = "EPIAS_processed.csv",
        gas_threshold_date: str = '2025-07-01',
        gas_price_before: float = 1127.82,
        gas_price_after: float = 1409.77
) -> pd.DataFrame:
    """
    Main function to process EPIAS energy market data.

    This function:
    1. Reads data from multiple CSV files
    2. Merges datasets on date and hour
    3. Converts data types
    4. Adds USD/TRY exchange rates
    5. Adds natural gas prices
    6. Removes unnecessary columns
    7. Saves the processed data

    Parameters:
    -----------
    ptf_path : str
        Path to PTF CSV file
    yuk_path : str
        Path to Load Forecast CSV file
    kgup_paths : List[str]
        List of paths to KGUP CSV files
    output_path : str
        Path for output CSV file
    gas_threshold_date : str
        Date threshold for natural gas price change
    gas_price_before : float
        Natural gas price before threshold
    gas_price_after : float
        Natural gas price after threshold

    Returns:
    --------
    pd.DataFrame
        Processed dataframe
    """
    # Default KGUP paths
    if kgup_paths is None:
        kgup_paths = [
            "data_s/Kesinlesmis_Gunluk_Uretim_Plani_(KGUP)-01012025-01042025.csv",
            "data_s/Kesinlesmis_Gunluk_Uretim_Plani_(KGUP)-02042025-02072025.csv",
            "data_s/Kesinlesmis_Gunluk_Uretim_Plani_(KGUP)-03072025-03102025.csv",
            "data_s/Kesinlesmis_Gunluk_Uretim_Plani_(KGUP)-03102025-30112025.csv"
        ]

    print("Step 1: Reading data files...")
    ptf_df, yuk_df, kgup_df = read_data_files(ptf_path, yuk_path, kgup_paths)

    print("Step 2: Merging dataframes...")
    df_final = merge_dataframes(ptf_df, yuk_df, kgup_df)

    print("Step 3: Converting data types...")
    df_final = convert_data_types(df_final)

    print("Step 4: Adding USD/TRY exchange rates...")
    start_date = df_final['Tarih'].min()
    end_date = df_final['Tarih'].max()
    df_final = add_usd_exchange_rate(df_final, start_date, end_date)

    print("Step 5: Adding natural gas prices...")
    df_final = add_natural_gas_prices(
        df_final,
        gas_threshold_date,
        gas_price_before,
        gas_price_after
    )

    print("Step 6: Removing unnecessary columns...")
    df_final = drop_unnecessary_columns(df_final)

    print(f"Step 7: Saving to {output_path}...")
    df_final.to_csv(output_path, index=False)

    print("Processing complete!")
    print(f"Final dataset shape: {df_final.shape}")

    return df_final


# Example usage
if __name__ == "__main__":
    df = process_epias_data()
    print("\nFirst few rows:")
    print(df.head())
    print("\nDataframe info:")
    print(df.info())
#---------------------------
