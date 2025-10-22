"""
Módulo de Machine Learning para CAF Dashboard
Incluye modelos de regresión, clasificación, clustering y reducción de dimensionalidad
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, MinMaxScaler, PowerTransformer
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, silhouette_score
)
import warnings
warnings.filterwarnings('ignore')

# Importar configuración
try:
    from config import THEME_TEMPLATE, PLOTLY_CONFIG
except ImportError:
    THEME_TEMPLATE = "plotly"
    PLOTLY_CONFIG = {}

def display_machine_learning_analysis(df: pd.DataFrame, numeric_cols: list, categorical_cols: list):
    """Análisis completo de Machine Learning"""
    
    st.header("🤖 Análisis de Machine Learning")
    
    if len(numeric_cols) == 0:
        st.warning("Se necesitan variables numéricas para análisis de ML")
        return
    
    # Configuración general
    st.subheader("⚙️ Configuración General")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("Tamaño del conjunto de prueba", 0.1, 0.4, 0.2, 0.05, key="ml_test_size")
    with col2:
        random_state = st.number_input("Semilla aleatoria", 0, 1000, 42, key="ml_random_state")
    with col3:
        cv_folds = st.selectbox("Pliegues de validación cruzada", [3, 5, 10], index=1, key="ml_cv_folds")
    
    # Selección de tipo de análisis
    analysis_type = st.selectbox(
        "Tipo de análisis",
        ["Regresión", "Clasificación", "Clustering", "Reducción de dimensionalidad", "Resumen Ejecutivo"],
        key="ml_analysis_type_selectbox"
    )
    
    if analysis_type == "Regresión":
        display_regression_analysis(df, numeric_cols, test_size, random_state, cv_folds)
    elif analysis_type == "Clasificación":
        display_classification_analysis(df, numeric_cols, categorical_cols, test_size, random_state, cv_folds)
    elif analysis_type == "Clustering":
        display_clustering_analysis(df, numeric_cols, test_size, random_state)
    elif analysis_type == "Reducción de dimensionalidad":
        display_dimensionality_reduction(df, numeric_cols, test_size, random_state)
    else:  # Resumen Ejecutivo
        display_ml_summary(df, numeric_cols, categorical_cols)

def display_regression_analysis(df: pd.DataFrame, numeric_cols: list, test_size: float, random_state: int, cv_folds: int):
    """Análisis de regresión"""
    
    st.subheader("📈 Análisis de Regresión")
    
    # Selección de variables
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Variables predictoras (X):**")
        feature_cols = st.multiselect(
            "Seleccionar variables predictoras",
            numeric_cols,
            default=numeric_cols[:min(3, len(numeric_cols))],
            key="ml_reg_features"
        )
    
    with col2:
        st.write("**Variable objetivo (y):**")
        target_col = st.selectbox(
            "Seleccionar variable objetivo",
            numeric_cols,
            key="ml_reg_target"
        )
    
    if len(feature_cols) < 1 or not target_col:
        st.warning("Selecciona al menos una variable predictora y una variable objetivo")
        return
    
    # Asegurar que la variable objetivo no esté en las predictoras
    if target_col in feature_cols:
        feature_cols = [col for col in feature_cols if col != target_col]
        st.info(f"Variable objetivo '{target_col}' removida de las predictoras")
    
    if len(feature_cols) < 1:
        st.warning("No hay variables predictoras después de remover la variable objetivo")
        return
    
    # Preparar datos
    all_cols = feature_cols + [target_col]
    df_clean = df[all_cols].dropna()
    
    if len(df_clean) < 10:
        st.error("No hay suficientes datos válidos para el análisis")
        return
    
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    
    # Verificar que y sea 1D
    if hasattr(y, 'ndim') and y.ndim > 1:
        y = y.iloc[:, 0] if hasattr(y, 'iloc') else y[:, 0]
        st.warning(f"Variable objetivo convertida a 1D. Forma original: {y.shape if hasattr(y, 'shape') else 'N/A'}")
    
    # Verificar que X y y tengan el formato correcto
    try:
        # Asegurar que y sea un array 1D de numpy
        if hasattr(y, 'values'):
            y = y.values
        if hasattr(y, 'ndim') and y.ndim > 1:
            y = y.ravel()
        
        # Asegurar que X sea un DataFrame o array 2D
        if hasattr(X, 'values'):
            X = X.values
        
        # Validar formas finales
        if X.ndim != 2:
            st.error(f"X debe ser 2D, pero tiene forma {X.shape}")
            return
        if y.ndim != 1:
            st.error(f"y debe ser 1D, pero tiene forma {y.shape}")
            return
        
        st.info(f"Formas de datos - X: {X.shape}, y: {y.shape}")
        
    except Exception as e:
        st.error(f"Error preparando datos: {str(e)}")
        return
    
    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Escalado
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Modelos de regresión
    st.write("**Modelos de Regresión:**")
    
    models = {
        "Regresión Lineal": "linear",
        "Ridge": "ridge", 
        "Lasso": "lasso",
        "Elastic Net": "elastic",
        "Random Forest": "rf",
        "Gradient Boosting": "gb",
        "SVR": "svr",
        "RANSAC": "ransac",
        "Theil-Sen": "theil"
    }
    
    selected_models = st.multiselect(
        "Seleccionar modelos a entrenar",
        list(models.keys()),
        default=["Regresión Lineal", "Random Forest", "Gradient Boosting"],
        key="ml_reg_models"
    )
    
    if st.button("🚀 Entrenar Modelos", key="train_reg"):
        train_regression_models(
            X_train_scaled, X_test_scaled, y_train, y_test,
            selected_models, cv_folds, random_state
        )

def train_regression_models(X_train, X_test, y_train, y_test, selected_models, cv_folds, random_state):
    """Entrena y evalúa modelos de regresión"""
    
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RANSACRegressor, TheilSenRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.svm import SVR
    
    results = []
    
    for model_name in selected_models:
        with st.spinner(f"Entrenando {model_name}..."):
            
            # Configurar modelo
            if model_name == "Regresión Lineal":
                model = LinearRegression()
            elif model_name == "Ridge":
                model = Ridge(alpha=1.0, random_state=random_state)
            elif model_name == "Lasso":
                model = Lasso(alpha=0.1, random_state=random_state)
            elif model_name == "Elastic Net":
                model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=random_state)
            elif model_name == "Random Forest":
                model = RandomForestRegressor(n_estimators=100, random_state=random_state)
            elif model_name == "Gradient Boosting":
                model = GradientBoostingRegressor(n_estimators=100, random_state=random_state)
            elif model_name == "SVR":
                model = SVR(kernel='rbf', C=1.0, gamma='scale')
            elif model_name == "RANSAC":
                model = RANSACRegressor(random_state=random_state)
            elif model_name == "Theil-Sen":
                model = TheilSenRegressor(random_state=random_state)
            
            # Entrenar modelo
            model.fit(X_train, y_train)
            
            # Predicciones
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Métricas
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            # Validación cruzada
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            results.append({
                'Modelo': model_name,
                'R² Entrenamiento': round(train_r2, 4),
                'R² Prueba': round(test_r2, 4),
                'RMSE Entrenamiento': round(train_rmse, 4),
                'RMSE Prueba': round(test_rmse, 4),
                'MAE Entrenamiento': round(train_mae, 4),
                'MAE Prueba': round(test_mae, 4),
                'CV R² (Media)': round(cv_mean, 4),
                'CV R² (Std)': round(cv_std, 4)
            })
    
    # Mostrar resultados
    if results:
        st.subheader("📊 Resultados de los Modelos")
        
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, width='stretch')
        
        # Gráfico de comparación
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('R² Score', 'RMSE', 'MAE', 'CV R²'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        models_list = results_df['Modelo'].tolist()
        
        # R² Score
        fig.add_trace(
            go.Bar(x=models_list, y=results_df['R² Prueba'], name='R² Prueba', marker_color='lightblue'),
            row=1, col=1
        )
        
        # RMSE
        fig.add_trace(
            go.Bar(x=models_list, y=results_df['RMSE Prueba'], name='RMSE Prueba', marker_color='lightcoral'),
            row=1, col=2
        )
        
        # MAE
        fig.add_trace(
            go.Bar(x=models_list, y=results_df['MAE Prueba'], name='MAE Prueba', marker_color='lightgreen'),
            row=2, col=1
        )
        
        # CV R²
        fig.add_trace(
            go.Bar(x=models_list, y=results_df['CV R² (Media)'], name='CV R²', marker_color='lightyellow'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False, template=THEME_TEMPLATE)
        st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)

def display_classification_analysis(df: pd.DataFrame, numeric_cols: list, categorical_cols: list, test_size: float, random_state: int, cv_folds: int):
    """Análisis de clasificación"""
    
    st.subheader("🎯 Análisis de Clasificación")
    
    # Crear variable objetivo si no hay categóricas
    if len(categorical_cols) == 0:
        st.info("No hay variables categóricas. Creando variable objetivo binaria...")
        
        target_col = st.selectbox("Variable para crear objetivo binario", numeric_cols, key="ml_class_binary_target")
        
        # Usar percentiles para asegurar balance de clases
        percentile_options = {
            "Mediana (50%)": 50,
            "Tercil inferior (33%)": 33,
            "Tercil superior (67%)": 67,
            "Cuartil inferior (25%)": 25,
            "Cuartil superior (75%)": 75
        }
        
        percentile_choice = st.selectbox("Método de binarización", list(percentile_options.keys()), key="ml_class_percentile")
        percentile = percentile_options[percentile_choice]
        
        threshold = np.percentile(df[target_col].dropna(), percentile)
        
        st.info(f"Umbral calculado: {threshold:.3f} (percentil {percentile}%)")
        
        df_analysis = df.copy()
        df_analysis['target'] = (df_analysis[target_col] > threshold).astype(int)
        target_col = 'target'
        
        # Verificar balance de clases
        class_counts = df_analysis['target'].value_counts()
        st.info(f"Distribución de clases: {dict(class_counts)}")
        
    else:
        target_col = st.selectbox("Variable objetivo categórica", categorical_cols, key="ml_class_cat_target")
        df_analysis = df.copy()
    
    # Selección de características
    feature_cols = st.multiselect(
        "Variables predictoras",
        numeric_cols,
        default=numeric_cols[:min(5, len(numeric_cols))],
        key="ml_class_features"
    )
    
    if len(feature_cols) < 1:
        st.warning("Selecciona al menos una variable predictora")
        return
    
    # Asegurar que la variable objetivo no esté en las predictoras
    if target_col in feature_cols:
        feature_cols = [col for col in feature_cols if col != target_col]
        st.info(f"Variable objetivo '{target_col}' removida de las predictoras")
    
    if len(feature_cols) < 1:
        st.warning("No hay variables predictoras después de remover la variable objetivo")
        return
    
    # Preparar datos
    all_cols = feature_cols + [target_col]
    df_clean = df_analysis[all_cols].dropna()
    
    if len(df_clean) < 10:
        st.error("No hay suficientes datos válidos para el análisis")
        return
    
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    
    # Codificar variable objetivo si es categórica
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
        class_names = le.classes_
    else:
        class_names = [str(i) for i in np.unique(y)]
    
    # Verificar que tenemos suficientes datos para cada clase
    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_class_count = np.min(class_counts)
    
    if min_class_count < 2:
        st.error(f"Error: Algunas clases tienen muy pocos miembros (mínimo: {min_class_count}). Se necesitan al menos 2 miembros por clase para clasificación.")
        st.info("Considera usar un análisis de regresión en su lugar, o ajustar el umbral para crear la variable objetivo.")
        return
    
    # Mostrar información sobre las clases
    st.info(f"Clases detectadas: {len(unique_classes)} clases con distribución: {dict(zip(class_names, class_counts))}")
    
    # División de datos
    # Verificar si podemos usar stratify
    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_class_count = np.min(class_counts)
    
    if min_class_count >= 2:
        # Usar stratify si todas las clases tienen al menos 2 miembros
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        # No usar stratify si alguna clase tiene menos de 2 miembros
        st.warning(f"Algunas clases tienen muy pocos miembros (mínimo: {min_class_count}). No se usará stratify.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    
    # Escalado
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Modelos de clasificación
    models = {
        "Regresión Logística": "logistic",
        "Random Forest": "rf",
        "Gradient Boosting": "gb", 
        "SVM": "svm",
        "Naive Bayes": "nb",
        "K-NN": "knn",
        "LDA": "lda",
        "QDA": "qda"
    }
    
    selected_models = st.multiselect(
        "Modelos de clasificación",
        list(models.keys()),
        default=["Regresión Logística", "Random Forest", "SVM"],
        key="class_models"
    )
    
    if st.button("🚀 Entrenar Modelos", key="train_class"):
        train_classification_models(
            X_train_scaled, X_test_scaled, y_train, y_test,
            selected_models, cv_folds, random_state, class_names
        )

def train_classification_models(X_train, X_test, y_train, y_test, selected_models, cv_folds, random_state, class_names):
    """Entrena y evalúa modelos de clasificación"""
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    
    results = []
    
    for model_name in selected_models:
        with st.spinner(f"Entrenando {model_name}..."):
            
            # Configurar modelo
            if model_name == "Regresión Logística":
                model = LogisticRegression(random_state=random_state, max_iter=1000)
            elif model_name == "Random Forest":
                model = RandomForestClassifier(n_estimators=100, random_state=random_state)
            elif model_name == "Gradient Boosting":
                model = GradientBoostingClassifier(n_estimators=100, random_state=random_state)
            elif model_name == "SVM":
                model = SVC(random_state=random_state)
            elif model_name == "Naive Bayes":
                model = GaussianNB()
            elif model_name == "K-NN":
                model = KNeighborsClassifier(n_neighbors=5)
            elif model_name == "LDA":
                model = LinearDiscriminantAnalysis()
            elif model_name == "QDA":
                model = QuadraticDiscriminantAnalysis()
            
            # Entrenar modelo
            model.fit(X_train, y_train)
            
            # Predicciones
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Métricas
            train_acc = accuracy_score(y_train, y_pred_train)
            test_acc = accuracy_score(y_test, y_pred_test)
            train_prec = precision_score(y_train, y_pred_train, average='weighted', zero_division=0)
            test_prec = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
            train_rec = recall_score(y_train, y_pred_train, average='weighted', zero_division=0)
            test_rec = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
            train_f1 = f1_score(y_train, y_pred_train, average='weighted', zero_division=0)
            test_f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
            
            # Validación cruzada
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            results.append({
                'Modelo': model_name,
                'Accuracy Entrenamiento': round(train_acc, 4),
                'Accuracy Prueba': round(test_acc, 4),
                'Precision Prueba': round(test_prec, 4),
                'Recall Prueba': round(test_rec, 4),
                'F1-Score Prueba': round(test_f1, 4),
                'CV Accuracy (Media)': round(cv_mean, 4),
                'CV Accuracy (Std)': round(cv_std, 4)
            })
    
    # Mostrar resultados
    if results:
        st.subheader("📊 Resultados de Clasificación")
        
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, width='stretch')
        
        # Matriz de confusión del mejor modelo
        best_model_idx = results_df['Accuracy Prueba'].idxmax()
        best_model_name = results_df.loc[best_model_idx, 'Modelo']
        
        st.subheader(f"🎯 Mejor Modelo: {best_model_name}")
        
        # Reentrenar el mejor modelo para mostrar matriz de confusión
        if best_model_name == "Regresión Logística":
            best_model = LogisticRegression(random_state=random_state, max_iter=1000)
        elif best_model_name == "Random Forest":
            best_model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        elif best_model_name == "Gradient Boosting":
            best_model = GradientBoostingClassifier(n_estimators=100, random_state=random_state)
        elif best_model_name == "SVM":
            best_model = SVC(random_state=random_state)
        elif best_model_name == "Naive Bayes":
            best_model = GaussianNB()
        elif best_model_name == "K-NN":
            best_model = KNeighborsClassifier(n_neighbors=5)
        elif best_model_name == "LDA":
            best_model = LinearDiscriminantAnalysis()
        elif best_model_name == "QDA":
            best_model = QuadraticDiscriminantAnalysis()
        
        best_model.fit(X_train, y_train)
        y_pred_best = best_model.predict(X_test)
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred_best)
        
        fig = px.imshow(
            cm, 
            text_auto=True, 
            aspect="auto",
            labels=dict(x="Predicción", y="Verdadero", color="Cantidad"),
            x=class_names,
            y=class_names,
            title=f"Matriz de Confusión - {best_model_name}"
        )
        fig.update_layout(template=THEME_TEMPLATE)
        st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)

def display_clustering_analysis(df: pd.DataFrame, numeric_cols: list, test_size: float, random_state: int):
    """Análisis de clustering"""
    
    st.subheader("🔍 Análisis de Clustering")
    
    # Selección de variables
    feature_cols = st.multiselect(
        "Variables para clustering",
        numeric_cols,
        default=numeric_cols[:min(5, len(numeric_cols))],
        key="ml_cluster_features"
    )
    
    if len(feature_cols) < 2:
        st.warning("Selecciona al menos 2 variables para clustering")
        return
    
    # Preparar datos
    df_clean = df[feature_cols].dropna()
    
    if len(df_clean) < 10:
        st.error("No hay suficientes datos válidos para el análisis")
        return
    
    X = df_clean[feature_cols]
    
    # Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Configuración de clustering
    col1, col2 = st.columns(2)
    
    with col1:
        n_clusters = st.slider("Número de clusters", 2, 10, 3, key="ml_n_clusters")
    
    with col2:
        clustering_method = st.selectbox(
            "Método de clustering",
            ["K-Means", "DBSCAN", "Agglomerative", "Spectral"],
            key="ml_clustering_method"
        )
    
    if st.button("🚀 Ejecutar Clustering", key="run_clustering"):
        run_clustering_analysis(X_scaled, X, n_clusters, clustering_method, random_state, feature_cols)

def run_clustering_analysis(X_scaled, X, n_clusters, method, random_state, feature_cols):
    """Ejecuta análisis de clustering"""
    
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
    
    # Verificar que tenemos suficientes datos para clustering
    if len(X_scaled) < n_clusters * 2:
        st.error(f"No hay suficientes datos para {n_clusters} clusters. Se necesitan al menos {n_clusters * 2} puntos de datos.")
        return
    
    with st.spinner(f"Ejecutando {method}..."):
        
        if method == "K-Means":
            clusterer = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        elif method == "DBSCAN":
            eps = st.slider("Parámetro eps para DBSCAN", 0.1, 2.0, 0.5, 0.1, key="ml_dbscan_eps")
            clusterer = DBSCAN(eps=eps, min_samples=5)
        elif method == "Agglomerative":
            linkage = st.selectbox("Tipo de linkage", ["ward", "complete", "average", "single"], key="ml_agglomerative_linkage")
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        elif method == "Spectral":
            clusterer = SpectralClustering(n_clusters=n_clusters, random_state=random_state)
        
        # Ejecutar clustering
        cluster_labels = clusterer.fit_predict(X_scaled)
        
        # Crear DataFrame con resultados
        df_results = X.copy()
        df_results['Cluster'] = cluster_labels
        
        # Mostrar estadísticas por cluster
        st.subheader("📊 Estadísticas por Cluster")
        
        cluster_stats = df_results.groupby('Cluster').agg({
            col: ['count', 'mean', 'std'] for col in feature_cols
        }).round(3)
        
        st.dataframe(cluster_stats, width='stretch')
        
        # Gráficos de clustering
        if len(feature_cols) >= 2:
            st.subheader("📈 Visualización de Clusters")
            
            # Gráfico 2D
            fig = px.scatter(
                df_results, 
                x=feature_cols[0], 
                y=feature_cols[1],
                color='Cluster',
                title=f"Clustering {method} - {feature_cols[0]} vs {feature_cols[1]}",
                labels={feature_cols[0]: feature_cols[0], feature_cols[1]: feature_cols[1]}
            )
            fig.update_layout(template=THEME_TEMPLATE)
            st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)
            
            # Si hay más de 2 variables, mostrar PCA
            if len(feature_cols) > 2:
                from sklearn.decomposition import PCA
                
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                
                df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
                df_pca['Cluster'] = cluster_labels
                
                fig_pca = px.scatter(
                    df_pca,
                    x='PC1',
                    y='PC2',
                    color='Cluster',
                    title=f"Clustering en espacio PCA - {method}",
                    labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var.)', 
                           'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var.)'}
                )
                fig_pca.update_layout(template=THEME_TEMPLATE)
                st.plotly_chart(fig_pca, width='stretch', config=PLOTLY_CONFIG)

def display_dimensionality_reduction(df: pd.DataFrame, numeric_cols: list, test_size: float, random_state: int):
    """Análisis de reducción de dimensionalidad"""
    
    st.subheader("📉 Reducción de Dimensionalidad")
    
    # Selección de variables
    feature_cols = st.multiselect(
        "Variables para reducción de dimensionalidad",
        numeric_cols,
        default=numeric_cols[:min(10, len(numeric_cols))],
        key="ml_dimred_features"
    )
    
    if len(feature_cols) < 3:
        st.warning("Selecciona al menos 3 variables para reducción de dimensionalidad")
        return
    
    # Preparar datos
    df_clean = df[feature_cols].dropna()
    
    if len(df_clean) < 10:
        st.error("No hay suficientes datos válidos para el análisis")
        return
    
    X = df_clean[feature_cols]
    
    # Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Métodos de reducción de dimensionalidad
    methods = {
        "PCA": "pca",
        "Factor Analysis": "fa",
        "t-SNE": "tsne",
        "UMAP": "umap",
        "MDS": "mds",
        "Isomap": "isomap"
    }
    
    selected_methods = st.multiselect(
        "Métodos de reducción de dimensionalidad",
        list(methods.keys()),
        default=["PCA", "t-SNE"],
        key="ml_dimred_methods"
    )
    
    if st.button("🚀 Ejecutar Reducción", key="run_dimred"):
        run_dimensionality_reduction(X_scaled, X, selected_methods, random_state, feature_cols)

def run_dimensionality_reduction(X_scaled, X, methods, random_state, feature_cols):
    """Ejecuta reducción de dimensionalidad"""
    
    from sklearn.decomposition import PCA, FactorAnalysis
    from sklearn.manifold import TSNE, MDS, Isomap
    import umap
    
    for method in methods:
        with st.spinner(f"Ejecutando {method}..."):
            
            st.subheader(f"🔬 {method}")
            
            if method == "PCA":
                # PCA con número óptimo de componentes
                pca = PCA()
                X_reduced = pca.fit_transform(X_scaled)
                
                # Gráfico de varianza explicada
                cumsum = np.cumsum(pca.explained_variance_ratio_)
                
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Varianza Explicada por Componente', 'Varianza Acumulada'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                fig.add_trace(
                    go.Bar(x=list(range(1, len(pca.explained_variance_ratio_)+1)), 
                          y=pca.explained_variance_ratio_,
                          name='Varianza Explicada'),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=list(range(1, len(cumsum)+1)), 
                              y=cumsum,
                              mode='lines+markers',
                              name='Varianza Acumulada'),
                    row=1, col=2
                )
                
                fig.update_layout(height=400, template=THEME_TEMPLATE)
                st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)
                
                # Usar las primeras 2 componentes para visualización
                X_2d = X_reduced[:, :2]
                
            elif method == "Factor Analysis":
                n_components = min(5, len(feature_cols) - 1)
                fa = FactorAnalysis(n_components=n_components, random_state=random_state)
                X_reduced = fa.fit_transform(X_scaled)
                X_2d = X_reduced[:, :2]
                
            elif method == "t-SNE":
                perplexity = min(30, len(X_scaled) // 4)
                tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
                X_2d = tsne.fit_transform(X_scaled)
                
            elif method == "UMAP":
                n_neighbors = min(15, len(X_scaled) // 4)
                umap_reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=random_state)
                X_2d = umap_reducer.fit_transform(X_scaled)
                
            elif method == "MDS":
                mds = MDS(n_components=2, random_state=random_state)
                X_2d = mds.fit_transform(X_scaled)
                
            elif method == "Isomap":
                n_neighbors = min(10, len(X_scaled) // 4)
                isomap = Isomap(n_components=2, n_neighbors=n_neighbors)
                X_2d = isomap.fit_transform(X_scaled)
            
            # Crear DataFrame para visualización
            df_reduced = pd.DataFrame(X_2d, columns=['Componente 1', 'Componente 2'])
            df_reduced.index = X.index
            
            # Gráfico de dispersión 2D
            fig = px.scatter(
                df_reduced,
                x='Componente 1',
                y='Componente 2',
                title=f"Reducción de Dimensionalidad - {method}",
                labels={'Componente 1': 'Componente 1', 'Componente 2': 'Componente 2'}
            )
            fig.update_layout(template=THEME_TEMPLATE)
            st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)
            
            # Mostrar información sobre la reducción
            if method == "PCA":
                st.info(f"Los primeros 2 componentes explican el {cumsum[1]:.1%} de la varianza total")
            else:
                st.info(f"Reducción de {len(feature_cols)} dimensiones a 2 dimensiones usando {method}")


def display_ml_summary(df: pd.DataFrame, numeric_cols: list, categorical_cols: list):
    """Resumen ejecutivo del análisis de Machine Learning"""
    
    st.subheader("📋 Resumen Ejecutivo de Machine Learning")
    
    # Generar resumen automático
    summary = generate_ml_summary(df, numeric_cols, categorical_cols)
    
    # Mostrar resumen en columnas
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Hallazgos Principales")
        for finding in summary['findings']:
            st.write(f"• {finding}")
    
    with col2:
        st.subheader("⚠️ Alertas y Recomendaciones")
        for alert in summary['alerts']:
            st.write(f"• {alert}")
    
    # Métricas clave
    st.subheader("📊 Métricas Clave")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Variables Numéricas", summary['n_numeric'])
    with col2:
        st.metric("Variables Categóricas", summary['n_categorical'])
    with col3:
        st.metric("Calidad de Datos", summary['data_quality'])
    with col4:
        st.metric("Recomendación", summary['recommendation'])
    
    # Análisis de viabilidad para ML
    st.subheader("🔍 Análisis de Viabilidad para ML")
    
    # Análisis de correlación
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        high_corr_pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:
                    high_corr_pairs.append((numeric_cols[i], numeric_cols[j], corr_val))
        
        if high_corr_pairs:
            st.warning(f"⚠️ Se encontraron {len(high_corr_pairs)} pares de variables altamente correlacionadas (>0.8)")
            with st.expander("Ver pares correlacionados"):
                for var1, var2, corr in high_corr_pairs:
                    st.write(f"• {var1} ↔ {var2}: {corr:.3f}")
        else:
            st.success("✅ No hay variables altamente correlacionadas")
    
    # Análisis de dimensionalidad
    n_obs = len(df)
    n_features = len(numeric_cols)
    
    if n_obs < n_features * 10:
        st.warning("⚠️ Pocas observaciones para el número de características (riesgo de sobreajuste)")
    elif n_obs < n_features * 5:
        st.error("❌ Muy pocas observaciones para el número de características")
    else:
        st.success("✅ Relación observaciones/características adecuada")
    
    # Recomendaciones de modelos
    st.subheader("🤖 Recomendaciones de Modelos")
    
    if summary['data_quality'] == "Excelente" and n_obs >= 100:
        st.success("✅ Datos de alta calidad - Se recomiendan modelos complejos")
        st.write("• **Regresión:** Random Forest, Gradient Boosting, XGBoost")
        st.write("• **Clasificación:** Random Forest, SVM, Neural Networks")
        st.write("• **Clustering:** K-Means, DBSCAN, Hierarchical")
    elif summary['data_quality'] in ["Excelente", "Buena"] and n_obs >= 50:
        st.info("ℹ️ Datos de buena calidad - Se recomiendan modelos intermedios")
        st.write("• **Regresión:** Linear Regression, Random Forest")
        st.write("• **Clasificación:** Logistic Regression, Random Forest")
        st.write("• **Clustering:** K-Means, Hierarchical")
    else:
        st.warning("⚠️ Datos limitados - Se recomiendan modelos simples")
        st.write("• **Regresión:** Linear Regression, Ridge, Lasso")
        st.write("• **Clasificación:** Logistic Regression, Naive Bayes")
        st.write("• **Clustering:** K-Means")


def generate_ml_summary(df: pd.DataFrame, numeric_cols: list, categorical_cols: list) -> dict:
    """Genera un resumen ejecutivo del análisis de ML"""
    findings = []
    alerts = []
    
    n_numeric = len(numeric_cols)
    n_categorical = len(categorical_cols)
    n_observations = len(df)
    
    # Análisis de calidad de datos
    missing_pct = df.isna().sum().sum() / (n_observations * (n_numeric + n_categorical)) * 100
    if missing_pct < 5:
        data_quality = "Excelente"
    elif missing_pct < 15:
        data_quality = "Buena"
    elif missing_pct < 30:
        data_quality = "Regular"
    else:
        data_quality = "Pobre"
    
    # Generar hallazgos
    findings.append(f"Dataset con {n_observations} observaciones")
    findings.append(f"{n_numeric} variables numéricas y {n_categorical} categóricas")
    findings.append(f"Calidad de datos: {data_quality}")
    
    # Análisis de dimensionalidad
    if n_observations >= n_numeric * 10:
        findings.append("Relación observaciones/características adecuada para ML")
    elif n_observations >= n_numeric * 5:
        findings.append("Relación observaciones/características aceptable")
    else:
        findings.append("Relación observaciones/características limitada")
    
    # Análisis de correlación
    if n_numeric > 1:
        corr_matrix = df[numeric_cols].corr()
        high_corr_count = 0
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corr_count += 1
        
        if high_corr_count > 0:
            findings.append(f"Se encontraron {high_corr_count} pares de variables altamente correlacionadas")
    
    # Generar alertas
    if missing_pct > 15:
        alerts.append(f"Alto porcentaje de valores faltantes ({missing_pct:.1f}%)")
    
    if n_observations < n_numeric * 5:
        alerts.append("Pocas observaciones para el número de características")
    
    if n_numeric == 0:
        alerts.append("No hay variables numéricas para análisis de ML")
    
    if n_observations < 30:
        alerts.append("Dataset muy pequeño para análisis de ML robusto")
    
    # Recomendación
    if data_quality == "Excelente" and n_observations >= 100:
        recommendation = "Excelente"
    elif data_quality in ["Excelente", "Buena"] and n_observations >= 50:
        recommendation = "Buena"
    elif data_quality in ["Excelente", "Buena", "Regular"]:
        recommendation = "Revisar"
    else:
        recommendation = "Requiere atención"
    
    return {
        'findings': findings,
        'alerts': alerts,
        'n_numeric': n_numeric,
        'n_categorical': n_categorical,
        'data_quality': data_quality,
        'recommendation': recommendation
    }
