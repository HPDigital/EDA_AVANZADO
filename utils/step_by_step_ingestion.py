"""
Sistema de ingesta paso a paso para datos
1. Ingesta individual de archivos
2. Reconocimiento y corrección de tipos
3. Selección de columnas pivote para unificar
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings

from .multi_format_loader import MultiFormatLoader
from .smart_ingestion import SmartCSVIngestion, TableInfo, ColumnInfo

warnings.filterwarnings('ignore')


@dataclass
class FileIngestionResult:
    """Resultado de la ingesta de un archivo individual"""
    file_path: str
    file_name: str
    file_format: str
    success: bool
    error_message: Optional[str] = None
    df: Optional[pd.DataFrame] = None
    table_info: Optional[TableInfo] = None
    column_types: Optional[Dict[str, str]] = None  # col_name -> detected_type
    corrected_types: Optional[Dict[str, str]] = None  # col_name -> corrected_type


class StepByStepIngestion:
    """Sistema de ingesta paso a paso"""
    
    def __init__(self):
        self.multi_loader = MultiFormatLoader()
        self.smart_ingestion = SmartCSVIngestion()
        self.ingested_files: List[FileIngestionResult] = []
        self.unified_df: Optional[pd.DataFrame] = None
    
    def step1_ingest_individual_files(self, file_paths: List[str]) -> List[FileIngestionResult]:
        """
        PASO 1: Ingesta individual de cada archivo sin unificar
        """
        results = []
        
        for file_path in file_paths:
            result = FileIngestionResult(
                file_path=file_path,
                file_name=Path(file_path).name,
                file_format=Path(file_path).suffix.lower(),
                success=False
            )
            
            try:
                # Cargar archivo con MultiFormatLoader
                df, metadata = self.multi_loader.load_file(file_path)
                
                if df is None or df.empty:
                    result.error_message = "Archivo vacío o no se pudo cargar"
                    results.append(result)
                    continue
                
                # Limpiar nombres de columnas
                df = self.smart_ingestion.clean_column_names(df)
                
                # Encontrar límites de la tabla
                start_row, end_row = self.smart_ingestion.find_table_boundaries(df)
                
                if start_row >= end_row:
                    result.error_message = "No se encontraron datos válidos en el archivo"
                    results.append(result)
                    continue
                
                # Extraer la tabla real
                table_df = df.iloc[start_row:end_row].copy()
                
                # Detectar tipos de columnas automáticamente
                column_types = {}
                columns_info = []
                time_columns = []
                
                for col in table_df.columns:
                    dtype, confidence = self.smart_ingestion.detect_column_type(table_df[col], col)
                    column_types[col] = dtype
                    
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
                
                # Crear TableInfo
                has_time_column = len(time_columns) > 0
                primary_time_column = time_columns[0] if time_columns else None
                
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
                    data_preview=table_df.copy(),
                    has_time_column=has_time_column,
                    time_column=primary_time_column
                )
                
                # Aplicar normalización básica según tipos detectados
                normalized_df = self._normalize_dataframe(table_df, column_types)
                
                result.success = True
                result.df = normalized_df
                result.table_info = table_info
                result.column_types = column_types
                result.corrected_types = column_types.copy()  # Inicialmente igual a detectados
                
            except Exception as e:
                result.error_message = str(e)
            
            results.append(result)
        
        self.ingested_files = results
        return results
    
    def step2_correct_column_types(self, file_index: int, column_corrections: Dict[str, str]) -> bool:
        """
        PASO 2: Corregir tipos de columnas de un archivo específico
        
        Args:
            file_index: Índice del archivo en self.ingested_files
            column_corrections: Dict con {col_name: new_type}
        """
        if file_index >= len(self.ingested_files):
            return False
        
        file_result = self.ingested_files[file_index]
        if not file_result.success:
            return False
        
        # Actualizar tipos corregidos
        file_result.corrected_types.update(column_corrections)
        
        # Re-normalizar el DataFrame con los tipos corregidos
        file_result.df = self._normalize_dataframe(file_result.df, file_result.corrected_types)
        
        # Actualizar TableInfo
        for col_info in file_result.table_info.columns:
            if col_info.name in file_result.corrected_types:
                col_info.dtype = file_result.corrected_types[col_info.name]
        
        return True
    
    def step3_unify_with_pivot_columns(self, pivot_columns: Dict[str, str]) -> pd.DataFrame:
        """
        PASO 3: Unificar archivos usando columnas pivote seleccionadas
        
        Args:
            pivot_columns: Dict con {file_index: column_name} para cada archivo
        """
        unified_dfs = []
        
        for file_index, file_result in enumerate(self.ingested_files):
            if not file_result.success:
                continue
            
            df = file_result.df.copy()
            
            # Agregar metadatos del archivo
            df['_source_file'] = file_result.file_name
            df['_file_index'] = file_index
            df['_file_format'] = file_result.file_format
            
            # Agregar información de tipos de columnas
            # Primero usar tipos corregidos si existen
            for col, col_type in file_result.corrected_types.items():
                if col in df.columns:
                    df[f'_type_{col}'] = col_type
            
            # Luego agregar tipos detectados automáticamente para columnas sin corrección
            for col in df.columns:
                if not col.startswith('_'):  # Solo columnas de datos
                    type_col = f'_type_{col}'
                    if type_col not in df.columns:
                        # Buscar tipo detectado en table_info.columns
                        col_type = 'qualitative'  # Por defecto
                        if file_result.table_info and file_result.table_info.columns:
                            for col_info in file_result.table_info.columns:
                                if col_info.name == col:
                                    col_type = col_info.dtype
                                    break
                        df[type_col] = col_type
            
            # Marcar columna pivote si está especificada
            if file_index in pivot_columns:
                pivot_col = pivot_columns[file_index]
                if pivot_col in df.columns:
                    df['_pivot_column'] = pivot_col
                    df['_pivot_value'] = df[pivot_col]
            
            unified_dfs.append(df)
        
        if unified_dfs:
            # Concatenar todos los DataFrames
            self.unified_df = pd.concat(unified_dfs, ignore_index=True, sort=False)
            
            # Ordenar por columna pivote si está disponible
            if '_pivot_value' in self.unified_df.columns:
                try:
                    self.unified_df = self.unified_df.sort_values('_pivot_value', na_position='last')
                except:
                    pass
            
            return self.unified_df
        
        return pd.DataFrame()
    
    def _normalize_dataframe(self, df: pd.DataFrame, column_types: Dict[str, str]) -> pd.DataFrame:
        """Normaliza un DataFrame según los tipos de columnas especificados"""
        normalized_df = df.copy()
        
        for col, col_type in column_types.items():
            if col not in normalized_df.columns:
                continue
            
            if col_type == 'quantitative':
                # Normalización para datos cuantitativos
                series_obj = normalized_df[col].astype(str)
                # Limpiar espacios duros y símbolos
                series_obj = series_obj.str.replace('\u00A0', ' ', regex=True)
                series_obj = series_obj.str.replace('[%$€]', '', regex=True)
                
                # Detectar si usa formato europeo (coma como decimal, punto como miles)
                # Si hay comas pero no puntos, es formato europeo
                has_comma = series_obj.str.contains(',').any()
                has_dot = series_obj.str.contains(r'\.').any()
                
                if has_comma and not has_dot:
                    # Formato europeo: 1.200,50 -> 1200.50
                    series_obj = series_obj.str.replace('.', '', regex=False)  # quitar miles
                    series_obj = series_obj.str.replace(',', '.', regex=False)  # decimal
                elif has_comma and has_dot:
                    # Formato mixto: determinar por posición
                    # Si la coma está después del punto, es decimal: 1200,50
                    # Si el punto está después de la coma, es miles: 1.200,50
                    def fix_mixed_format(val):
                        if ',' in val and '.' in val:
                            comma_pos = val.rfind(',')
                            dot_pos = val.rfind('.')
                            if comma_pos > dot_pos:
                                # Coma es decimal: 1200,50 -> 1200.50
                                return val.replace(',', '.')
                            else:
                                # Punto es decimal: 1.200,50 -> 1200.50
                                return val.replace('.', '').replace(',', '.')
                        return val
                    series_obj = series_obj.apply(fix_mixed_format)
                # Si no hay comas, mantener como está (formato americano)
                
                normalized_df[col] = pd.to_numeric(series_obj, errors='coerce')
                
            elif col_type == 'datetime':
                # Normalización para fechas
                normalized_df[col] = pd.to_datetime(normalized_df[col], errors='coerce', infer_datetime_format=True)
                
            elif col_type == 'qualitative':
                # Normalización para datos cualitativos
                normalized_df[col] = normalized_df[col].astype(str)
                # Limpiar valores nulos
                normalized_df[col] = normalized_df[col].replace(['nan', 'None', 'null'], np.nan)
        
        return normalized_df
    
    def get_file_summary(self, file_index: int) -> Dict[str, Any]:
        """Obtiene un resumen de un archivo específico"""
        if file_index >= len(self.ingested_files):
            return {}
        
        file_result = self.ingested_files[file_index]
        
        if not file_result.success:
            return {
                'success': False,
                'error': file_result.error_message,
                'file_name': file_result.file_name
            }
        
        summary = {
            'success': True,
            'file_name': file_result.file_name,
            'file_format': file_result.file_format,
            'rows': len(file_result.df),
            'columns': len(file_result.df.columns),
            'column_info': []
        }
        
        for col in file_result.df.columns:
            if col.startswith('_'):
                continue
                
            col_type = file_result.corrected_types.get(col, 'unknown')
            summary['column_info'].append({
                'name': col,
                'type': col_type,
                'null_count': int(file_result.df[col].isna().sum()),
                'unique_count': int(file_result.df[col].nunique()),
                'sample_values': file_result.df[col].head(3).tolist()
            })
        
        return summary
    
    def get_all_files_summary(self) -> List[Dict[str, Any]]:
        """Obtiene un resumen de todos los archivos"""
        summaries = []
        for i in range(len(self.ingested_files)):
            summaries.append(self.get_file_summary(i))
        return summaries
    
    def get_available_columns_for_pivot(self, file_index: int) -> List[str]:
        """Obtiene las columnas disponibles para usar como pivote en un archivo"""
        if file_index >= len(self.ingested_files):
            return []
        
        file_result = self.ingested_files[file_index]
        if not file_result.success:
            return []
        
        # Retornar columnas que no sean metadatos
        return [col for col in file_result.df.columns if not col.startswith('_')]
    
    def get_unified_dataframe(self) -> pd.DataFrame:
        """Retorna el DataFrame unificado si existe"""
        return self.unified_df if self.unified_df is not None else pd.DataFrame()
    
    def reset(self):
        """Reinicia el sistema de ingesta"""
        try:
            print("[CLEANUP] Limpiando sistema de ingesta...")
            
            # Limpiar archivos ingeridos
            for file_result in self.ingested_files:
                if hasattr(file_result, 'df') and file_result.df is not None:
                    del file_result.df
                if hasattr(file_result, 'table_info') and file_result.table_info is not None:
                    if hasattr(file_result.table_info, 'data_preview'):
                        del file_result.table_info.data_preview
                if hasattr(file_result, 'column_types'):
                    del file_result.column_types
                if hasattr(file_result, 'corrected_types'):
                    del file_result.corrected_types
            
            self.ingested_files = []
            
            # Limpiar DataFrame unificado
            if self.unified_df is not None:
                del self.unified_df
                self.unified_df = None
            
            # Limpiar referencias a otros objetos
            if hasattr(self, 'multi_loader'):
                del self.multi_loader
            
            if hasattr(self, 'smart_ingestion'):
                del self.smart_ingestion
            
            # Reinicializar objetos necesarios
            self.multi_loader = MultiFormatLoader()
            self.smart_ingestion = SmartCSVIngestion()
            
            # Forzar garbage collection múltiple
            import gc
            for _ in range(3):
                gc.collect()
            
            print("[CLEANUP] Sistema de ingesta limpiado")
            
        except Exception as e:
            print(f"[ERROR] Error en reset: {str(e)}")
            # Limpiar de todas formas
            self.ingested_files = []
            self.unified_df = None
