"""
Sistema de ingesta inteligente de datos CSV
Detecta automáticamente archivos, tablas, encabezados y tipos de datos
"""
import os
import re
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

# Detectar si Streamlit está disponible
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False
    class st:
        """Dummy Streamlit para compatibilidad"""
        @staticmethod
        def warning(msg): print(f"Warning: {msg}")
        @staticmethod
        def error(msg): print(f"Error: {msg}")
        @staticmethod
        def spinner(msg): return DummyContext()
        @staticmethod
        def progress(val): return DummyProgress()

class DummyContext:
    def __enter__(self): return self
    def __exit__(self, *args): pass

class DummyProgress:
    def progress(self, val): pass


@dataclass
class ColumnInfo:
    """Información de una columna detectada"""
    name: str
    dtype: str  # 'quantitative', 'qualitative', 'datetime'
    confidence: float
    sample_values: List[str]
    null_count: int
    unique_count: int


@dataclass
class TableInfo:
    """Información de una tabla detectada en un archivo"""
    file_path: str
    table_name: str
    sheet_name: Optional[str]
    start_row: int
    end_row: int
    columns: List[ColumnInfo]
    time_columns: List[str]
    row_count: int
    column_count: int
    file_format: str
    encoding: str
    separator: str
    additional_info: Dict
    data_preview: pd.DataFrame
    has_time_column: bool = False
    time_column: Optional[str] = None


class SmartCSVIngestion:
    """Sistema inteligente de ingesta de archivos CSV"""
    
    def __init__(self):
        self.time_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # DD/MM/YYYY
            r'\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY
            r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
            r'\d{4}年\d{1,2}月\d{1,2}日',  # Chinese format
            r'\d{1,2}/\d{1,2}/\d{4}',  # M/D/YYYY
            r'\d{4}-\d{1,2}-\d{1,2}',  # YYYY-M-D
            r'\d{2}/\d{1,2}/\d{4}',  # DD/M/YYYY
            r'\d{1,2}/\d{2}/\d{4}',  # D/MM/YYYY
            r'\d{4}\.\d{2}\.\d{2}',  # YYYY.MM.DD
            r'\d{2}\.\d{2}\.\d{4}',  # DD.MM.YYYY
            r'\d{4} \d{2} \d{2}',  # YYYY MM DD
            r'\d{2} \d{2} \d{4}',  # DD MM YYYY
        ]
        
        self.time_keywords = [
            'date', 'fecha', 'time', 'tiempo', 'year', 'año', 'month', 'mes',
            'day', 'día', 'timestamp', 'datetime', 'period', 'período',
            'référence', 'reference', 'année', 'mois', 'jour',
            'created', 'updated', 'modified', 'created_at', 'updated_at',
            'fecha_nacimiento', 'birth_date', 'fecha_inicio', 'start_date',
            'fecha_fin', 'end_date', 'fecha_creacion', 'creation_date'
        ]
        
        self.quantitative_indicators = [
            'nombre', 'number', 'count', 'total', 'sum', 'amount', 'value',
            'quantity', 'volume', 'rate', 'percentage', 'ratio', 'index',
            'score', 'nbre', 'nb', 'montant', 'valeur', 'age', 'edad', 
            'id', 'code', 'numero', 'num', 'rank', 'position', 'ordre', 'order',
            'poblacion', 'population', 'habitants', 'resident', 'residents',
            'ingreso', 'income', 'salario', 'salary', 'precio', 'price',
            'cost', 'costo', 'gasto', 'expense', 'peso', 'weight', 'altura', 'height'
        ]

    def find_data_files(self, data_dir: str) -> List[str]:
        """Encuentra todos los archivos de datos soportados en el directorio"""
        from .multi_format_loader import MultiFormatLoader
        
        data_files = []
        data_path = Path(data_dir)
        
        if not data_path.exists():
            return data_files
        
        # Obtener formatos soportados
        supported_formats = MultiFormatLoader().get_supported_formats()
        
        # Buscar archivos con extensiones soportadas
        for extension in supported_formats.keys():
            for file_path in data_path.rglob(f"*{extension}"):
                data_files.append(str(file_path))
        
        return sorted(data_files)
    
    def find_csv_files(self, data_dir: str) -> List[str]:
        """Encuentra todos los archivos CSV en el directorio (compatibilidad)"""
        csv_files = []
        data_path = Path(data_dir)
        
        if not data_path.exists():
            return csv_files
            
        for file_path in data_path.rglob("*.csv"):
            csv_files.append(str(file_path))
            
        return sorted(csv_files)

    def detect_encoding(self, file_path: str) -> str:
        """Detecta la codificación del archivo"""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read(1000)  # Leer solo una muestra
                return encoding
            except UnicodeDecodeError:
                continue
                
        return 'utf-8'  # Fallback

    def detect_separator(self, file_path: str, encoding: str) -> str:
        """Detecta el separador del CSV"""
        separators = [';', ',', '\t', '|']
        
        with open(file_path, 'r', encoding=encoding) as f:
            sample = f.read(1000)
            
        for sep in separators:
            if sample.count(sep) > sample.count(',') * 0.5:
                return sep
                
        return ','  # Fallback

    def find_table_boundaries(self, df: pd.DataFrame) -> Tuple[int, int]:
        """Encuentra los límites reales de la tabla en el DataFrame"""
        # Buscar la primera fila con datos válidos
        start_row = 0
        for i, row in df.iterrows():
            if not row.isna().all() and len(str(row.iloc[0])) > 0:
                start_row = i
                break
        
        # Buscar la última fila con datos válidos
        end_row = len(df)
        for i in range(len(df) - 1, -1, -1):
            if not df.iloc[i].isna().all():
                end_row = i + 1
                break
                
        return start_row, end_row

    def _normalize_column_name(self, column_name: str) -> str:
        """Normaliza el nombre de una columna para mejor detección"""
        # Reemplazar guiones bajos y guiones con espacios
        normalized = column_name.lower().replace('_', ' ').replace('-', ' ')
        # Limpiar espacios múltiples
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized

    def detect_column_type(self, series: pd.Series, column_name: str) -> Tuple[str, float]:
        """Detecta el tipo de una columna con nivel de confianza"""
        series_clean = series.dropna()
        
        if len(series_clean) == 0:
            return 'qualitative', 0.0
        
        # Normalizar nombre de columna para mejor detección
        normalized_name = self._normalize_column_name(column_name)
        
        # PRIMERO: Verificar si es una variable temporal (años)
        time_confidence = self._check_time_variable(series_clean, normalized_name)
        if time_confidence > 0.5:  # Reducido el umbral para mejor detección
            return 'datetime', time_confidence
        
        # SEGUNDO: Verificar si es datetime (fechas completas)
        datetime_confidence = self._check_datetime(series_clean, normalized_name)
        if datetime_confidence > 0.6:  # Reducido el umbral
            return 'datetime', datetime_confidence
        
        # TERCERO: Verificar si es cuantitativo
        quantitative_confidence = self._check_quantitative(series_clean, normalized_name)
        if quantitative_confidence > 0.5:  # Reducido el umbral
            return 'quantitative', quantitative_confidence
        
        # Por defecto, cualitativo
        return 'qualitative', 0.8

    def _check_datetime(self, series: pd.Series, column_name: str) -> float:
        """Verifica si una serie es de tipo datetime"""
        confidence = 0.0
        
        # Verificar nombre de columna
        col_lower = column_name.lower()
        for keyword in self.time_keywords:
            if keyword in col_lower:
                confidence += 0.4  # Aumentado el peso del nombre
                break
        
        # Verificar patrones en los datos (muestra más grande)
        sample_size = min(20, len(series))
        sample_values = series.head(sample_size).astype(str)
        datetime_matches = 0
        
        for value in sample_values:
            value_str = str(value).strip()
            if value_str and value_str != 'nan':
                # Intentar parsear como fecha con múltiples formatos
                try:
                    # Intentar diferentes inferencias de fecha
                    pd.to_datetime(value_str, infer_datetime_format=True)
                    datetime_matches += 1
                    continue
                except:
                    try:
                        # Intentar con format específico común
                        pd.to_datetime(value_str, format='%Y-%m-%d')
                        datetime_matches += 1
                        continue
                    except:
                        try:
                            pd.to_datetime(value_str, format='%d/%m/%Y')
                            datetime_matches += 1
                            continue
                        except:
                            pass
                
                # Verificar patrones regex
                for pattern in self.time_patterns:
                    if re.search(pattern, value_str):
                        datetime_matches += 1
                        break
        
        if len(sample_values) > 0:
            confidence += (datetime_matches / len(sample_values)) * 0.6
            
        return min(confidence, 1.0)

    def _check_time_variable(self, series: pd.Series, column_name: str) -> float:
        """Verifica si una serie es una variable temporal (años, períodos)"""
        confidence = 0.0
        
        # El nombre ya viene normalizado desde detect_column_type
        time_indicators = [
            'year', 'año', 'année', 'period', 'período', 'période',
            'date', 'fecha', 'time', 'tiempo', 'temps', 'nombre', 'personnes'
        ]
        
        # Buscar indicadores temporales en el nombre
        for indicator in time_indicators:
            if indicator in column_name:
                confidence += 0.4  # Alto peso por el nombre
                break
        
        # Verificar patrones específicos para columnas con guiones bajos
        time_patterns = [
            r'.*year.*', r'.*año.*', r'.*date.*', r'.*fecha.*',
            r'.*time.*', r'.*period.*', r'.*nombre.*personnes.*'
        ]
        
        for pattern in time_patterns:
            if re.search(pattern, column_name):
                confidence += 0.3
                break
        
        # Verificar si los valores son años válidos
        year_count = 0
        total_count = len(series)
        
        # Muestra para verificación
        sample_size = min(50, total_count)
        sample_series = series.head(sample_size) if sample_size > 0 else series
        
        for value in sample_series:
            try:
                # Intentar convertir a número
                clean_value = str(value).replace(',', '.').replace(' ', '').strip()
                num_value = float(clean_value)
                
                # Verificar si es un año válido (1900-2100)
                if 1900 <= num_value <= 2100:
                    year_count += 1
                # Verificar si es un período (ej: 2023Q1, 2023-01)
                elif re.match(r'^\d{4}[Qq]\d{1}$', clean_value) or re.match(r'^\d{4}-\d{2}$', clean_value):
                    year_count += 1
                    
            except (ValueError, TypeError):
                pass
        
        if len(sample_series) > 0:
            year_ratio = year_count / len(sample_series)
            confidence += year_ratio * 0.6
            
        # Bonus si todos los valores son años consecutivos o cercanos (serie temporal)
        if year_count > 0:
            try:
                numeric_values = []
                for value in sample_series:
                    try:
                        clean_value = str(value).replace(',', '.').replace(' ', '').strip()
                        num_value = float(clean_value)
                        if 1900 <= num_value <= 2100:
                            numeric_values.append(num_value)
                    except:
                        pass
                
                if len(numeric_values) >= 3:
                    # Verificar si son años consecutivos o con patrón temporal
                    sorted_values = sorted(numeric_values)
                    differences = [sorted_values[i+1] - sorted_values[i] for i in range(len(sorted_values)-1)]
                    
                    # Si las diferencias son consistentes (1 año, 2 años, etc.)
                    if all(diff >= 1 and diff <= 5 for diff in differences):
                        confidence += 0.2  # Bonus por patrón temporal
                        
            except:
                pass
        
        return min(confidence, 1.0)

    def _check_quantitative(self, series: pd.Series, column_name: str) -> float:
        """Verifica si una serie es cuantitativa (NO temporal)"""
        confidence = 0.0
        
        # EXCLUIR variables temporales de la detección cuantitativa
        time_indicators = ['year', 'año', 'année', 'period', 'período', 'période', 'date', 'fecha', 'time', 'tiempo']
        
        for indicator in time_indicators:
            if indicator in column_name:
                return 0.0  # No es cuantitativa si es temporal
        
        # Verificar nombre de columna para indicadores cuantitativos
        for keyword in self.quantitative_indicators:
            if keyword in column_name:
                confidence += 0.4  # Aumentado el peso del nombre
                break
        
        # Verificar si los valores son numéricos
        numeric_count = 0
        total_count = len(series)
        
        # Muestra más grande para mejor detección
        sample_size = min(100, total_count)  # Aumentado el tamaño de muestra
        sample_series = series.head(sample_size) if sample_size > 0 else series
        
        for value in sample_series:
            try:
                # Limpiar y convertir a float
                clean_value = str(value).replace(',', '.').replace(' ', '').replace('%', '').replace('$', '').replace('€', '')
                float_val = float(clean_value)
                
                # EXCLUIR años de la detección cuantitativa (más estricto)
                if (1900 <= float_val <= 2100 and len(clean_value) == 4) or float_val == 0:
                    continue  # Es probablemente un año o cero, no lo contamos como cuantitativo
                
                # Para otros valores numéricos
                if not pd.isna(float_val) and np.isfinite(float_val):
                    numeric_count += 1
                    
            except (ValueError, TypeError):
                pass
        
        if len(sample_series) > 0:
            numeric_ratio = numeric_count / len(sample_series)
            confidence += numeric_ratio * 0.6
            
        # Bonus si la serie ya es de tipo numérico en pandas
        if pd.api.types.is_numeric_dtype(series):
            confidence += 0.2
            
        return min(confidence, 1.0)

    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpia los nombres de las columnas"""
        df_clean = df.copy()
        
        new_columns = []
        for col in df_clean.columns:
            # Limpiar espacios y caracteres especiales
            clean_col = str(col).strip()
            clean_col = re.sub(r'[^\w\s]', '_', clean_col)
            clean_col = re.sub(r'\s+', '_', clean_col)
            clean_col = clean_col.lower()
            
            # Si está vacío, darle un nombre genérico
            if not clean_col or clean_col == 'nan':
                clean_col = f'column_{len(new_columns) + 1}'
                
            new_columns.append(clean_col)
        
        df_clean.columns = new_columns
        return df_clean

    def process_data_file(self, file_path: str) -> List[TableInfo]:
        """Procesa un archivo de datos (cualquier formato soportado) y detecta tablas"""
        from .multi_format_loader import MultiFormatLoader
        
        tables = []
        
        try:
            # Usar el cargador multi-formato
            loader = MultiFormatLoader()
            df, metadata = loader.load_file(file_path)
            
            if df is None or df.empty:
                return tables
            
            # Limpiar nombres de columnas
            df = self.clean_column_names(df)
            
            # Encontrar límites de la tabla
            start_row, end_row = self.find_table_boundaries(df)
            
            if start_row >= end_row:
                return tables
            
            # Extraer la tabla real
            table_df = df.iloc[start_row:end_row].copy()
            
            # Detectar información de columnas
            columns_info = []
            time_columns = []
            
            for col in table_df.columns:
                dtype, confidence = self.detect_column_type(table_df[col], col)
                
                col_info = ColumnInfo(
                    name=col,
                    dtype=dtype,
                    confidence=confidence,
                    sample_values=table_df[col].head(5).astype(str).tolist(),
                    null_count=int(table_df[col].isna().sum()),
                    unique_count=int(table_df[col].nunique())
                )
                
                columns_info.append(col_info)
                
                if dtype == 'datetime':
                    time_columns.append(col)
            
            # Determinar columnas temporales
            has_time_column = len(time_columns) > 0
            primary_time_column = time_columns[0] if time_columns else None
            
            # Crear información de la tabla
            table_info = TableInfo(
                file_path=file_path,
                table_name=Path(file_path).stem,
                sheet_name=None,
                start_row=start_row,
                end_row=end_row,
                columns=columns_info,
                time_columns=time_columns,
                row_count=len(table_df),
                column_count=len(table_df.columns),
                file_format=metadata.get('format', 'unknown'),
                encoding=metadata.get('encoding', 'unknown'),
                separator=metadata.get('separator', 'unknown'),
                additional_info=metadata,
                data_preview=table_df.copy(),  # Agregar preview de datos
                has_time_column=has_time_column,
                time_column=primary_time_column
            )
            
            tables.append(table_info)
            
        except Exception as e:
            if HAS_STREAMLIT:
                st.error(f"Error procesando archivo {file_path}: {str(e)}")
            else:
                print(f"Error procesando archivo {file_path}: {str(e)}")
        
        return tables

    def process_csv_file(self, file_path: str) -> List[TableInfo]:
        """Procesa un archivo CSV y detecta tablas (compatibilidad)"""
        tables = []
        
        try:
            encoding = self.detect_encoding(file_path)
            separator = self.detect_separator(file_path, encoding)
            
            # Leer el archivo con diferentes configuraciones si es necesario
            df = None
            for sep in [separator, ';', ',', '\t']:
                try:
                    df = pd.read_csv(file_path, sep=sep, encoding=encoding, 
                                   na_values=['', ' ', 'NA', 'N/A', 'NULL'])
                    break
                except Exception:
                    continue
            
            if df is None or df.empty:
                return tables
            
            # Limpiar nombres de columnas
            df = self.clean_column_names(df)
            
            # Encontrar límites de la tabla
            start_row, end_row = self.find_table_boundaries(df)
            
            if start_row >= end_row:
                return tables
            
            # Extraer la tabla real
            table_df = df.iloc[start_row:end_row].copy()
            
            # Detectar información de columnas
            columns_info = []
            time_columns = []
            
            for col in table_df.columns:
                dtype, confidence = self.detect_column_type(table_df[col], col)
                
                col_info = ColumnInfo(
                    name=col,
                    dtype=dtype,
                    confidence=confidence,
                    sample_values=table_df[col].head(5).astype(str).tolist(),
                    null_count=int(table_df[col].isna().sum()),
                    unique_count=int(table_df[col].nunique())
                )
                columns_info.append(col_info)
                
                if dtype == 'datetime':
                    time_columns.append(col)
            
            # Determinar columnas temporales
            has_time_column = len(time_columns) > 0
            primary_time_column = time_columns[0] if time_columns else None
            
            # Crear información de la tabla
            table_info = TableInfo(
                file_path=file_path,
                table_name=Path(file_path).stem,
                sheet_name=None,
                start_row=start_row,
                end_row=end_row,
                columns=columns_info,
                time_columns=time_columns,
                row_count=len(table_df),
                column_count=len(table_df.columns),
                file_format='csv',
                encoding=encoding,
                separator=separator,
                additional_info={'method': 'csv_legacy'},
                data_preview=table_df.copy(),
                has_time_column=has_time_column,
                time_column=primary_time_column
            )
            
            tables.append(table_info)
            
        except Exception as e:
            st.warning(f"Error procesando {file_path}: {str(e)}")
            
        return tables

    def unify_tables_by_time(self, tables: List[TableInfo]) -> pd.DataFrame:
        """Unifica las tablas por variable de tiempo"""
        if not tables:
            return pd.DataFrame()
        
        unified_dfs = []
        
        for table in tables:
            df = table.data_preview.copy()
            
            # Agregar metadatos
            df['_source_file'] = Path(table.file_path).name
            df['_table_index'] = 0  # Por ahora solo una tabla por archivo
            
            # Procesar columnas de tiempo automáticamente
            time_columns_found = []
            for col in df.columns:
                col_lower = col.lower()
                
                # Verificar si es una columna temporal
                if any(keyword in col_lower for keyword in self.time_keywords):
                    try:
                        # Para años, mantener como entero pero marcarlo como temporal
                        if 'year' in col_lower or 'año' in col_lower or 'année' in col_lower:
                            # Convertir a entero si es posible
                            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                            time_columns_found.append(col)
                        else:
                            # Para fechas completas, convertir a datetime
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                            time_columns_found.append(col)
                    except:
                        pass
            
            df['_time_columns'] = ','.join(time_columns_found) if time_columns_found else None
            
            # Agregar información de tipos de columnas
            for col_info in table.columns:
                df[f'_type_{col_info.name}'] = col_info.dtype

            # Normalización fuerte según el tipo detectado
            # - quantitative -> numérico (manejo de comas, símbolos y espacios)
            # - datetime -> conversión a fecha real
            for col_info in table.columns:
                col = col_info.name
                if col not in df.columns:
                    continue
                if col_info.dtype == 'quantitative':
                    # Reemplazar separadores no imprimibles y símbolos comunes
                    series_obj = df[col].astype(str).str.replace('\\u00A0', ' ', regex=True)
                    series_obj = series_obj.str.replace('[%$€]', '', regex=True)

                    # Detectar patrón de separadores en una muestra
                    sample = series_obj.dropna().head(50)
                    comma_count = sample.str.contains(',', regex=False).sum()
                    dot_count = sample.str.contains(r'\.', regex=False).sum()

                    if comma_count and dot_count:
                        # Si ambos existen, intentar inferir decimal por la última ocurrencia típica
                        # Heurística: si hay más comas que puntos, asumir formato europeo
                        european = comma_count >= dot_count
                    elif comma_count:
                        european = True
                    elif dot_count:
                        european = False
                    else:
                        european = False

                    if european:
                        # Quitar separadores de miles con punto y usar coma como decimal
                        cleaned = series_obj.str.replace('.', '', regex=False)
                        cleaned = cleaned.str.replace(',', '.', regex=False)
                    else:
                        # Quitar separadores de miles con coma y usar punto como decimal
                        cleaned = series_obj.str.replace(',', '', regex=False)

                    df[col] = pd.to_numeric(cleaned, errors='coerce')
                elif col_info.dtype == 'datetime':
                    df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
            
            unified_dfs.append(df)
        
        if unified_dfs:
            # Concatenar todos los DataFrames
            result_df = pd.concat(unified_dfs, ignore_index=True, sort=False)
            
            # Ordenar por tiempo si es posible
            time_cols = [col for col in result_df.columns if any(keyword in col.lower() for keyword in self.time_keywords)]
            if time_cols:
                try:
                    # Usar la primera columna temporal encontrada
                    primary_time_col = time_cols[0]
                    result_df = result_df.sort_values(primary_time_col, na_position='last')
                except:
                    pass
            
            return result_df
        
        return pd.DataFrame()

    def process_directory(self, data_dir: str) -> Dict[str, Any]:
        """Procesa todo el directorio de datos"""
        data_files = self.find_data_files(data_dir)
        
        if not data_files:
            return {
                'success': False,
                'message': f"No se encontraron archivos de datos soportados en {data_dir}",
                'files': [],
                'files_processed': 0,
                'tables': [],
                'unified_df': pd.DataFrame()
            }
        
        all_tables = []
        
        if HAS_STREAMLIT:
            with st.spinner(f"Procesando {len(data_files)} archivos de datos..."):
                progress_bar = st.progress(0)

                for i, file_path in enumerate(data_files):
                    try:
                        tables = self.process_data_file(file_path)
                        all_tables.extend(tables)
                        progress_bar.progress((i + 1) / len(data_files))
                    except Exception as e:
                        st.error(f"Error procesando {file_path}: {str(e)}")
        else:
            # Modo Tkinter sin UI de progreso
            for i, file_path in enumerate(data_files):
                try:
                    print(f"Procesando {i+1}/{len(data_files)}: {Path(file_path).name}")
                    tables = self.process_data_file(file_path)
                    all_tables.extend(tables)
                except Exception as e:
                    print(f"Error procesando {file_path}: {str(e)}")

        # Unificar tablas
        unified_df = self.unify_tables_by_time(all_tables)

        return {
            'success': True,
            'message': f"Procesados {len(data_files)} archivos, detectadas {len(all_tables)} tablas",
            'files': data_files,
            'files_processed': len(data_files),
            'tables': all_tables,
            'unified_df': unified_df
        }

    def get_data_summary(self, tables: List[TableInfo]) -> Dict[str, any]:
        """Genera un resumen de los datos detectados"""
        summary = {
            'total_files': len(set(table.file_path for table in tables)),
            'total_tables': len(tables),
            'total_columns': sum(len(table.columns) for table in tables),
            'files_with_time': len([t for t in tables if t.has_time_column]),
            'column_types': {
                'quantitative': 0,
                'qualitative': 0,
                'datetime': 0
            }
        }
        
        for table in tables:
            for col in table.columns:
                summary['column_types'][col.dtype] += 1
        
        return summary


def display_ingestion_results(results: Dict[str, any]):
    """Muestra los resultados de la ingesta en Streamlit"""
    if not results['success']:
        st.error(results['message'])
        return
    
    st.success(results['message'])
    
    # Mostrar archivos procesados
    with st.expander(f"📁 Archivos procesados ({len(results['files'])})"):
        for file_path in results['files']:
            st.write(f"• {Path(file_path).name}")
    
    # Mostrar resumen de tablas
    summary = SmartCSVIngestion().get_data_summary(results['tables'])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Archivos", summary['total_files'])
    with col2:
        st.metric("Tablas", summary['total_tables'])
    with col3:
        st.metric("Columnas", summary['total_columns'])
    with col4:
        st.metric("Con tiempo", summary['files_with_time'])
    
    # Mostrar tipos de columnas
    col_types = summary['column_types']
    st.write("**Tipos de columnas detectadas:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cuantitativas", col_types['quantitative'])
    with col2:
        st.metric("Cualitativas", col_types['qualitative'])
    with col3:
        st.metric("Tiempo", col_types['datetime'])
    
    # Mostrar preview de datos unificados
    if not results['unified_df'].empty:
        with st.expander("📊 Vista previa de datos unificados"):
            clean_df = clean_dataframe_for_streamlit(results['unified_df'])
            st.dataframe(clean_df.head(20), width='stretch')
            
            # Mostrar información de columnas
            st.write("**Información de columnas:**")
            col_info_df = pd.DataFrame({
                'Columna': results['unified_df'].columns,
                'Tipo': [str(results['unified_df'][col].dtype) for col in results['unified_df'].columns],
                'No nulos': [int(results['unified_df'][col].notna().sum()) for col in results['unified_df'].columns],
                'Únicos': [int(results['unified_df'][col].nunique()) for col in results['unified_df'].columns]
            })
            st.dataframe(col_info_df, width='stretch')


def clean_dataframe_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia el DataFrame para compatibilidad con Streamlit/Arrow"""
    
    df_clean = df.copy()
    
    # Convertir columnas que contienen listas mixtas a string
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            # Verificar si la columna contiene listas mixtas
            has_lists = df_clean[col].apply(lambda x: isinstance(x, list)).any()
            has_non_lists = df_clean[col].apply(lambda x: not isinstance(x, list) and pd.notna(x)).any()
            
            if has_lists and has_non_lists:
                # Convertir todo a string
                df_clean[col] = df_clean[col].astype(str)
    
    return df_clean


def create_data_quality_report(unified_df: pd.DataFrame) -> pd.DataFrame:
    """Crea un reporte de calidad de datos"""
    if unified_df.empty:
        return pd.DataFrame()
    
    quality_report = []
    
    for col in unified_df.columns:
        if col.startswith('_'):  # Skip metadata columns
            continue
            
        series = unified_df[col]
        
        # Asegurar que los valores sean compatibles con Arrow
        def safe_float(val):
            try:
                return float(val) if not pd.isna(val) else None
            except:
                return None
        
        def safe_int(val):
            try:
                return int(val) if not pd.isna(val) else None
            except:
                return None
        
        report_row = {
            'Columna': str(col),
            'Tipo': str(series.dtype),
            'Total': safe_int(len(series)),
            'No nulos': safe_int(series.notna().sum()),
            'Nulos': safe_int(series.isna().sum()),
            'Únicos': safe_int(series.nunique()),
            'Duplicados': safe_int(len(series) - series.nunique()),
            'Completitud (%)': safe_float((series.notna().sum() / len(series)) * 100)
        }
        
        # Estadísticas específicas por tipo
        if pd.api.types.is_numeric_dtype(series):
            clean_series = pd.to_numeric(series, errors='coerce').dropna()
            if len(clean_series) > 0:
                report_row.update({
                    'Mín': safe_float(clean_series.min()),
                    'Máx': safe_float(clean_series.max()),
                    'Media': safe_float(clean_series.mean()),
                    'Mediana': safe_float(clean_series.median()),
                    'Desv. Std': safe_float(clean_series.std())
                })
        
        quality_report.append(report_row)
    
    # Crear DataFrame y asegurar tipos compatibles
    df_report = pd.DataFrame(quality_report)
    
    # Convertir columnas numéricas a tipos nativos de Python
    for col in ['Total', 'No nulos', 'Nulos', 'Únicos', 'Duplicados']:
        if col in df_report.columns:
            df_report[col] = df_report[col].astype('Int64')  # Nullable integer
    
    for col in ['Completitud (%)', 'Mín', 'Máx', 'Media', 'Mediana', 'Desv. Std']:
        if col in df_report.columns:
            df_report[col] = df_report[col].astype('Float64')  # Nullable float
    
    return df_report
