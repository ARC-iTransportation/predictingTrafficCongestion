import pandas as pd
class normalization:
    def minmax_norm(df: pd.DataFrame) -> pd.DataFrame:
        """
        Min-Max Normalization
        Args:
            df: DataFrame: The DataFrame to be normalized.
        Returns:
            DataFrame: The normalized DataFrame.
        """
        return (df - df.min()) / (df.max() - df.min())
    
    def zscore_norm(df: pd.DataFrame) -> pd.DataFrame:
        """
        Z-Score Normalization
        Args:
            df: DataFrame: The DataFrame to be normalized.
        Returns:
            DataFrame: The normalized DataFrame.
        """
        return (df - df.mean()) / df.std()
    
    def decimal_scaling(df: pd.DataFrame) -> pd.DataFrame:
        """
        Decimal Scaling Normalization
        Args:
            df: DataFrame: The DataFrame to be normalized.
        Returns:
            DataFrame: The normalized DataFrame.
        """
        return df / 10 ** len(str(df.max()))