"""
Utilidades para carga y preparación de datos CAF
"""
import os
import pandas as pd
from config import PRESTATION_MAP, AGE_MAP, CAF_AGE_FILE, CAF_FAM_FILE


class CAFDataLoader:
    """Clase para manejar la carga de datos CAF"""
    
    @staticmethod
    def load_caf_csv(path: str) -> pd.DataFrame:
        """Carga un CSV CAF con manejo automático de separadores y decimales"""
        # Intentar diferentes configuraciones
        try:
            df = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8")
        except:
            try:
                df = pd.read_csv(path, sep=";", decimal=".", encoding="utf-8")
            except:
                df = pd.read_csv(path, sep=",", decimal=".", encoding="utf-8")
        
        # Limpiar nombres de columnas
        df.columns = [str(col).strip() for col in df.columns]
        
        # Procesar fechas
        if "Date référence" in df.columns:
            df["Date référence"] = pd.to_datetime(df["Date référence"], errors="coerce")
            df["Year"] = df["Date référence"].dt.year
        
        # Mapear prestaciones
        if "RSA ou PPA" in df.columns:
            df["RSA ou PPA"] = df["RSA ou PPA"].map(PRESTATION_MAP).fillna(df["RSA ou PPA"])
        
        # Mapear edades
        if "Age responsable dossier" in df.columns:
            df["Age responsable dossier"] = df["Age responsable dossier"].map(AGE_MAP).fillna(
                df["Age responsable dossier"]
            )
        
        return df
    
    @staticmethod
    def load_and_merge_caf_data(data_dir: str) -> pd.DataFrame:
        """Carga y combina los archivos de edad y familia"""
        age_path = os.path.join(data_dir, CAF_AGE_FILE)
        fam_path = os.path.join(data_dir, CAF_FAM_FILE)
        
        if not os.path.exists(age_path) or not os.path.exists(fam_path):
            raise FileNotFoundError(f"Archivos no encontrados en {data_dir}")
        
        df_age = CAFDataLoader.load_caf_csv(age_path)
        df_fam = CAFDataLoader.load_caf_csv(fam_path)
        
        df_age["Source"] = "Age"
        df_fam["Source"] = "Situation_Familiale"
        
        return pd.concat([df_age, df_fam], ignore_index=True, sort=False)
    
    @staticmethod
    def caf_to_long_format(df: pd.DataFrame) -> pd.DataFrame:
        """Convierte datos CAF a formato largo"""
        long_data = []
        
        for _, row in df.iterrows():
            base_info = {
                "Date": row.get("Date référence"),
                "Département": row.get("Nom département", ""),
                "Région": row.get("Nom région", ""),
                "Prestation": row.get("RSA ou PPA", ""),
                "Year": row.get("Year"),
                "Source": row.get("Source", "")
            }
            
            # Categoría según fuente
            if row.get("Source") == "Age":
                base_info["Category"] = f"Age_{row.get('Age responsable dossier', '')}"
            elif row.get("Source") == "Situation_Familiale":
                base_info["Category"] = f"Famille_{row.get('Situation familiale', '')}"
            else:
                base_info["Category"] = "Other"
            
            # Agregar métricas
            for metric in ["Nombre foyers RSA_PPA", "Nombre personnes RSA_PPA"]:
                if metric in row and pd.notna(row[metric]):
                    metric_data = base_info.copy()
                    metric_data["Indicator"] = metric
                    metric_data["Value"] = float(row[metric])
                    long_data.append(metric_data)
        
        return pd.DataFrame(long_data)
    
    @staticmethod
    def create_indicator_panel(long_df: pd.DataFrame) -> pd.DataFrame:
        """Crea panel de indicadores desde formato largo"""
        long_df["Indicator_Key"] = long_df["Category"] + "_" + long_df["Indicator"]
        
        panel = long_df.pivot_table(
            index=["Département", "Year"],
            columns="Indicator_Key",
            values="Value",
            aggfunc="sum"
        ).fillna(0)
        
        return panel.reset_index()


class DataPreprocessor:
    """Clase para preprocesamiento de datos"""
    
    @staticmethod
    def filter_by_coverage(df: pd.DataFrame, numeric_cols: list, 
                          min_coverage: float = 0.6) -> list:
        """Filtra columnas por cobertura mínima"""
        coverage = df[numeric_cols].notna().mean()
        return coverage[coverage >= min_coverage].index.tolist()
    
    @staticmethod
    def prepare_X(df: pd.DataFrame, cols: list, 
                  impute: str = "Media", 
                  scale: str = "Z-score", 
                  log1p: bool = False):
        """Prepara matriz X para análisis multivariado"""
        from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
        import numpy as np
        
        X = df[cols].copy()
        
        # Transformación log1p
        if log1p:
            for c in cols:
                col = pd.to_numeric(X[c], errors="coerce")
                if np.all(np.isfinite(col)) and (col.min() >= 0):
                    X[c] = np.log1p(col)
        
        # Imputación
        if impute == "Mediana":
            X_imp = X.apply(lambda s: pd.to_numeric(s, errors="coerce")).fillna(
                X.median(numeric_only=True)
            )
        else:
            X_imp = X.apply(lambda s: pd.to_numeric(s, errors="coerce")).fillna(
                X.mean(numeric_only=True)
            )
        
        # Escalado
        if scale == "Z-score":
            scaler = StandardScaler()
        elif scale == "Robusto (IQR)":
            scaler = RobustScaler()
        elif scale == "Min-Max":
            scaler = MinMaxScaler()
        else:
            return X_imp, X_imp.values, np.array(cols)
        
        X_scaled = scaler.fit_transform(X_imp.values)
        return X_imp, X_scaled, np.array(cols)