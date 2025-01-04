import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Session state'i başlat
if 'step' not in st.session_state:
    st.session_state.step = 1  # Başlangıç adımı

# Step 1: Load Dataset
if st.session_state.step >= 1:
    st.title("Model Evaluation and Comparison")
    st.write("## Step 1: Load Dataset")
    data = pd.read_csv('AmesHousing.csv')
    st.write("Dataset Loaded. Here's a preview:")
    st.dataframe(data.head())

    if st.button("Confirm Dataset Loading"):
        st.session_state.step = 2  # Bir sonraki adıma geç

# Step 2: Data Cleaning and Preprocessing
if st.session_state.step >= 2:
    st.write("## Step 2: Data Cleaning and Preprocessing")

    # Checkbox for handling missing values
    st.write("### Handle Missing Values")
    handle_missing_values = st.checkbox("Handle Missing Values", value=True)

    if handle_missing_values:
        numerical_cols = data.select_dtypes(include=np.number).columns
        categorical_cols = data.select_dtypes(include=['object']).columns

        # Impute missing values for numerical features using median
        imputer_numerical = SimpleImputer(strategy='median')
        data[numerical_cols] = imputer_numerical.fit_transform(data[numerical_cols])

        # Impute missing values for categorical features using most frequent
        imputer_categorical = SimpleImputer(strategy='most_frequent')
        data[categorical_cols] = imputer_categorical.fit_transform(data[categorical_cols])

    # Checkbox for encoding categorical variables
    st.write("### Encode Categorical Variables")
    encode_categorical = st.checkbox("Encode Categorical Variables", value=True)

    if encode_categorical:
        for col in categorical_cols:
            encoder = LabelEncoder()
            data[col] = encoder.fit_transform(data[col])

    # Checkbox for scaling/normalizing features
    st.write("### Scale/Normalize Features")
    scale_features = st.checkbox("Scale/Normalize Features", value=True)

    if scale_features:
        scaler = StandardScaler()
        data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    else:
        data_scaled = data

    st.write("Data after preprocessing:")
    st.dataframe(data_scaled.head())

    if st.button("Confirm Data Preprocessing"):
        st.session_state.step = 3  # Bir sonraki adıma geç

# Step 3: Unsupervised Learning for Feature Engineering
if st.session_state.step >= 3:
    st.write("## Step 3: Unsupervised Learning for Feature Engineering")

    # Checkbox for applying PCA
    st.write("### Apply PCA")
    apply_pca = st.checkbox("Apply PCA", value=True)

    if apply_pca:
        n_components = st.slider("Number of PCA Components", min_value=1, max_value=20, value=10)
        pca = PCA(n_components=n_components)
        data_pca = pca.fit_transform(data_scaled)
        st.write(f"PCA applied with {n_components} components.")
    else:
        data_pca = data_scaled

    if st.button("Confirm Feature Engineering"):
        st.session_state.step = 4  # Bir sonraki adıma geç

# Step 4: Split Data
if st.session_state.step >= 4:
    st.write("## Step 4: Split Data")
    X = data_pca
    y = data_scaled['SalePrice']  # Target variable

    # Slider for test size
    test_size = st.slider("Test Size", min_value=0.1, max_value=0.5, value=0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    st.write(f"Data split into training and testing sets with test size {test_size}.")

    if st.button("Confirm Data Splitting"):
        st.session_state.step = 5  # Bir sonraki adıma geç

# Step 5: Determine Models
if st.session_state.step >= 5:
    st.write("## Step 5: Determine Models")

    # Checkboxes for selecting models
    st.write("### Select Models")
    models_to_use = {
        'Linear Regression': st.checkbox("Linear Regression", value=True),
        'Decision Tree': st.checkbox("Decision Tree", value=True),
        'Gradient Boosting': st.checkbox("Gradient Boosting", value=True)
    }

    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(),
        'Gradient Boosting': GradientBoostingRegressor()
    }

    param_grids = {
        'Linear Regression': {},  # No hyperparameters to tune
        'Decision Tree': {
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10]
        },
        'Gradient Boosting': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    }

    # Filter models based on user selection
    selected_models = {model_name: models[model_name] for model_name, selected in models_to_use.items() if selected}
    selected_param_grids = {model_name: param_grids[model_name] for model_name, selected in models_to_use.items() if selected}

    st.write("The following models and parameter grids will be used:")
    st.json(selected_param_grids)

    if st.button("Confirm Model Selection"):
        st.session_state.step = 6  # Bir sonraki adıma geç

# Step 6: Pre-Tuning Results
if st.session_state.step >= 6:
    st.write("## Step 6: Pre-Tuning Results")
    pre_tuning_metrics = {}
    for model_name, model in selected_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = {
            'R2 Score': r2_score(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred)
        }
        pre_tuning_metrics[model_name] = metrics

    pre_tuning_df = pd.DataFrame(pre_tuning_metrics).T
    st.write("Metrics before tuning:")
    st.dataframe(pre_tuning_df)

    if st.button("Confirm Pre-Tuning Results"):
        st.session_state.step = 7  # Bir sonraki adıma geç

# Step 7: Hyperparameter Tuning
if st.session_state.step >= 7:
    st.write("## Step 7: Hyperparameter Tuning")

    # Eğitilmiş modelleri ve sonuçları session_state'de sakla
    if 'best_models' not in st.session_state:
        st.session_state.best_models = {}
    if 'post_tuning_metrics' not in st.session_state:
        st.session_state.post_tuning_metrics = {}

    # Toplam model sayısı
    total_models = len(selected_param_grids)

    # Her model için işlem yap
    for i, (model_name, params) in enumerate(selected_param_grids.items()):
        st.write(f"### Tuning {model_name} ({i + 1}/{total_models})")

        # Model ve parametreleri al
        model = selected_models[model_name]

        # Gradient Boosting için özel ilerleme çubuğu
        if model_name == "Gradient Boosting":
            # İlerleme çubuğu ve durum metni için session_state kullan
            if f"{model_name}_progress" not in st.session_state:
                st.session_state[f"{model_name}_progress"] = 0
                st.session_state[f"{model_name}_status"] = f"Tuning {model_name} - 0% complete"

            # İlerleme çubuğu ve durum metni
            progress_bar = st.progress(st.session_state[f"{model_name}_progress"])
            status_text = st.empty()
            status_text.text(st.session_state[f"{model_name}_status"])

            # GridSearchCV için özel bir callback fonksiyonu
            def update_progress(progress):
                st.session_state[f"{model_name}_progress"] = progress
                st.session_state[f"{model_name}_status"] = f"Tuning {model_name} - {int(progress * 100)}% complete"
                progress_bar.progress(st.session_state[f"{model_name}_progress"])
                status_text.text(st.session_state[f"{model_name}_status"])

            # GridSearchCV ile hiperparametre optimizasyonu
            if f"{model_name}_best_model" not in st.session_state:
                grid_search = GridSearchCV(
                    model,
                    params,
                    cv=5,
                    scoring='r2',
                    n_jobs=-1  # Tüm CPU çekirdeklerini kullan
                )

                # GridSearchCV'yi çalıştır
                grid_search.fit(X_train, y_train)

                # En iyi modeli session_state'de sakla
                st.session_state[f"{model_name}_best_model"] = grid_search.best_estimator_

                # İlerleme çubuğunu tamamla
                update_progress(1.0)
                st.session_state[f"{model_name}_status"] = f"{model_name} tuning completed!"
                status_text.text(st.session_state[f"{model_name}_status"])

            # En iyi modeli session_state'den al
            best_model = st.session_state[f"{model_name}_best_model"]

        else:
            # Diğer modeller için GridSearchCV
            if params:  # Hiperparametre optimizasyonu yapılacak
                if f"{model_name}_best_model" not in st.session_state:
                    grid_search = GridSearchCV(model, params, cv=5, scoring='r2')
                    grid_search.fit(X_train, y_train)
                    st.session_state[f"{model_name}_best_model"] = grid_search.best_estimator_

                # En iyi modeli session_state'den al
                best_model = st.session_state[f"{model_name}_best_model"]
            else:
                best_model = model

            st.session_state.best_models[model_name] = best_model

        # Modeli değerlendir
        if model_name not in st.session_state.post_tuning_metrics:
            y_pred = best_model.predict(X_test)
            metrics = {
                'R2 Score': r2_score(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'MAE': mean_absolute_error(y_test, y_pred)
            }
            st.session_state.post_tuning_metrics[model_name] = metrics

    # Tüm modellerin metriklerini göster
    post_tuning_df = pd.DataFrame(st.session_state.post_tuning_metrics).T
    st.write("Metrics after tuning:")
    st.dataframe(post_tuning_df)

    if st.button("Confirm Hyperparameter Tuning"):
        st.session_state.step = 8  # Bir sonraki adıma geç

# Step 8: Combine Results
if st.session_state.step >= 8:
    st.write("## Step 8: Combine Results")

    # Önceden hesaplanmış sonuçları kullan
    pre_tuning_df['Stage'] = 'Pre-Tuning'
    post_tuning_df = pd.DataFrame(st.session_state.post_tuning_metrics).T
    post_tuning_df['Stage'] = 'Post-Tuning'
    results_comparison = pd.concat([pre_tuning_df, post_tuning_df])
    st.write("Combined results:")
    st.dataframe(results_comparison)

    if st.button("Confirm Combined Results"):
        st.session_state.step = 9  # Bir sonraki adıma geç

# Step 9: Visualizations
if st.session_state.step >= 9:
    st.write("## Step 9: Visualizations")

    # Önceden hesaplanmış sonuçları kullan
    results_comparison = pd.concat([pre_tuning_df, post_tuning_df])

    # Visualization: R2 Score Comparison
    st.write("### R2 Score Comparison")
    fig_r2, ax_r2 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=results_comparison.reset_index(), x='index', y='R2 Score', hue='Stage', ax=ax_r2)
    ax_r2.set_title("R2 Score Comparison Before and After Tuning")
    ax_r2.set_ylabel("R2 Score")
    ax_r2.set_xlabel("Model")
    st.pyplot(fig_r2)

    # Visualization: RMSE Comparison
    st.write("### RMSE Comparison")
    fig_rmse, ax_rmse = plt.subplots(figsize=(10, 6))
    sns.barplot(data=results_comparison.reset_index(), x='index', y='RMSE', hue='Stage', ax=ax_rmse)
    ax_rmse.set_title("RMSE Comparison Before and After Tuning")
    ax_rmse.set_ylabel("Root Mean Squared Error (RMSE)")
    ax_rmse.set_xlabel("Model")
    st.pyplot(fig_rmse)

    # Visualization: MAE Comparison
    st.write("### MAE Comparison")
    fig_mae, ax_mae = plt.subplots(figsize=(10, 6))
    sns.barplot(data=results_comparison.reset_index(), x='index', y='MAE', hue='Stage', ax=ax_mae)
    ax_mae.set_title("MAE Comparison Before and After Tuning")
    ax_mae.set_ylabel("Mean Absolute Error (MAE)")
    ax_mae.set_xlabel("Model")
    st.pyplot(fig_mae)
