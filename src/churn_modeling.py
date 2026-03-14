from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.modules.model_io import save_best_model
from src.modules.ops_store import log_model_registration

matplotlib.use("Agg")
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)

RANDOM_STATE = 42
TEST_SIZE = 0.20

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT_DIR / "data" / "trusted" / "train.parquet"
ENGINEERED_DATASET_PATH = ROOT_DIR / "data" / "refined" / "train_engineered.parquet"
ARTIFACTS_DIR = ROOT_DIR / "artifacts" / "modeling"

FEATURE_NOTES = {
    "avg_seconds_per_call": "Regra de negocio: mede a intensidade media de cada ligacao. Ajuda a separar clientes que usam pouco a rede daqueles que fazem menos chamadas, mas chamadas mais longas.",
    "sms_per_call": "Regra de negocio: compara o uso de SMS com o volume de chamadas. Ajuda a identificar perfis mais orientados a mensagem do que a voz tradicional.",
    "calls_per_contact": "Regra de negocio: mede o quanto o cliente concentra chamadas em poucos numeros. Valores altos indicam rede de contato mais concentrada.",
    "contact_diversity_ratio": "Regra de negocio: mede a diversidade de contatos em relacao ao volume de chamadas. Ajuda a diferenciar clientes com rede social mais ampla.",
    "call_failure_rate": "Regra de negocio: resume a friccao operacional das chamadas. Se a taxa de falha sobe, existe sinal de experiencia ruim e maior propensao a churn.",
    "usage_per_month": "Regra de negocio: normaliza o tempo de uso pela antiguidade do cliente. Evita superestimar clientes antigos apenas por acumularem mais consumo.",
    "calls_per_month": "Regra de negocio: normaliza a frequencia de chamadas pelo tempo de base. Ajuda a capturar intensidade recorrente de uso.",
    "sms_per_month": "Regra de negocio: normaliza o uso de SMS pela antiguidade. Permite comparar perfis com tempos de relacionamento diferentes.",
    "value_per_month": "Regra de negocio: aproxima a monetizacao media mensal do cliente. Ajuda a entender valor economico recorrente.",
    "value_per_call": "Regra de negocio: relaciona valor gerado com o volume de chamadas. Ajuda a separar clientes que geram mais valor mesmo sem alto volume.",
    "complaint_status_interaction": "Regra de negocio: combina reclamacao com status operacional. O objetivo e capturar quando insatisfacao e contexto operacional ruim aparecem juntos.",
    "charge_value_gap": "Regra de negocio: mede a diferenca entre cobranca nominal e valor percebido ou capturado. Pode sinalizar desalinhamento economico.",
    "tenure_age_ratio": "Regra de negocio: relaciona tempo de base com idade do cliente. Funciona como proxy de ciclo de vida e maturidade de relacionamento.",
    "usage_value_efficiency": "Regra de negocio: compara o quanto de uso e entregue por unidade de valor. Pode sinalizar percepcao de custo-beneficio.",
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    eps = 1e-6

    # Ratios expose behavior more clearly than raw totals for churn.
    data["avg_seconds_per_call"] = data["seconds_of_use"] / (data["frequency_of_use"] + eps)
    data["sms_per_call"] = data["frequency_of_sms"] / (data["frequency_of_use"] + eps)
    data["calls_per_contact"] = data["frequency_of_use"] / (
        data["distinct_called_numbers"] + eps
    )
    data["contact_diversity_ratio"] = data["distinct_called_numbers"] / (
        data["frequency_of_use"] + eps
    )
    data["call_failure_rate"] = data["call_failure"] / (
        data["frequency_of_use"] + data["call_failure"] + eps
    )
    data["usage_per_month"] = data["seconds_of_use"] / (data["subscription_length"] + eps)
    data["calls_per_month"] = data["frequency_of_use"] / (data["subscription_length"] + eps)
    data["sms_per_month"] = data["frequency_of_sms"] / (data["subscription_length"] + eps)
    data["value_per_month"] = data["customer_value"] / (data["subscription_length"] + eps)
    data["value_per_call"] = data["customer_value"] / (data["frequency_of_use"] + eps)
    data["complaint_status_interaction"] = data["complains"] * data["status"]
    data["charge_value_gap"] = data["customer_value"] - data["charge_amount"]
    data["tenure_age_ratio"] = data["subscription_length"] / (data["age"] + eps)
    data["usage_value_efficiency"] = data["seconds_of_use"] / (data["customer_value"] + eps)

    return data


def build_preprocessors(feature_names: list[str]) -> tuple[ColumnTransformer, ColumnTransformer]:
    # Linear models benefit from scaling; tree models do not need it.
    scaled = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_names,
            )
        ]
    )
    tree = ColumnTransformer(
        [
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                feature_names,
            )
        ]
    )
    return scaled, tree


def build_models(feature_names: list[str], scale_pos_weight: float) -> dict[str, Pipeline]:
    scaled_preprocessor, tree_preprocessor = build_preprocessors(feature_names)

    # Use a mixed benchmark: linear, bagging, boosting, and gradient tree libraries.
    return {
        "Logistic Regression": Pipeline(
            [
                ("prep", scaled_preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2_000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("prep", tree_preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        n_jobs=1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Extra Trees": Pipeline(
            [
                ("prep", tree_preprocessor),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=500,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        n_jobs=1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("prep", tree_preprocessor),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=250,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("prep", tree_preprocessor),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=350,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                        scale_pos_weight=scale_pos_weight,
                        eval_metric="logloss",
                        n_jobs=1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "LightGBM": Pipeline(
            [
                ("prep", tree_preprocessor),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=350,
                        learning_rate=0.05,
                        num_leaves=31,
                        subsample=0.9,
                        colsample_bytree=0.8,
                        class_weight="balanced",
                        n_jobs=1,
                        random_state=RANDOM_STATE,
                        verbosity=-1,
                    ),
                ),
            ]
        ),
    }


def evaluate_with_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    models: dict[str, Pipeline],
) -> pd.DataFrame:
    # Stratification preserves the churn rate across folds.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "recall": "recall",
        "precision": "precision",
        "f1": "f1",
        "balanced_accuracy": "balanced_accuracy",
    }

    rows: list[dict[str, float | str]] = []

    for name, pipeline in models.items():
        # Keep evaluation homogeneous so model ranking is comparable.
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=1)
        row: dict[str, float | str] = {"model": name}
        for metric in scoring:
            row[f"cv_{metric}_mean"] = float(scores[f"test_{metric}"].mean())
            row[f"cv_{metric}_std"] = float(scores[f"test_{metric}"].std())
        rows.append(row)

    return pd.DataFrame(rows).sort_values("cv_roc_auc_mean", ascending=False).reset_index(
        drop=True
    )


def evaluate_on_holdout(
    X: pd.DataFrame,
    y: pd.Series,
    models: dict[str, Pipeline],
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    # A final holdout simulates model behavior on unseen data after benchmark selection.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    holdout_rows: list[dict[str, float | str]] = []
    probabilities: dict[str, np.ndarray] = {}

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        # Keep probabilities for thresholding, ROC/PR, and comparison plots.
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.50).astype(int)
        probabilities[name] = y_proba
        holdout_rows.append(
            {
                "model": name,
                "test_accuracy": float(accuracy_score(y_test, y_pred)),
                "test_balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                "test_precision": float(precision_score(y_test, y_pred)),
                "test_recall": float(recall_score(y_test, y_pred)),
                "test_f1": float(f1_score(y_test, y_pred)),
                "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
                "test_average_precision": float(average_precision_score(y_test, y_proba)),
            }
        )

    return (
        pd.DataFrame(holdout_rows).sort_values("test_roc_auc", ascending=False).reset_index(
            drop=True
        ),
        probabilities,
        X_train,
        y_train,
        X_test,
        y_test,
    )


def compare_raw_vs_engineered(raw_df: pd.DataFrame, engineered_df: pd.DataFrame) -> pd.DataFrame:
    results: list[dict[str, float | str]] = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "f1": "f1",
    }

    for label, dataset in (("raw_features", raw_df), ("engineered_features", engineered_df)):
        # Fix the model family to isolate the effect of feature engineering.
        X = dataset.drop(columns=["churn"])
        y = dataset["churn"].astype(int)
        scale_pos_weight = (y == 0).sum() / (y == 1).sum()
        model = build_models(X.columns.tolist(), scale_pos_weight)["XGBoost"]
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
        results.append(
            {
                "feature_set": label,
                "roc_auc": float(scores["test_roc_auc"].mean()),
                "average_precision": float(scores["test_average_precision"].mean()),
                "f1": float(scores["test_f1"].mean()),
            }
        )

    return pd.DataFrame(results)


def build_threshold_table(
    y_true: pd.Series,
    y_proba: np.ndarray,
) -> tuple[pd.DataFrame, float]:
    # Threshold analysis is useful when recall and precision have different business costs.
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    valid_precision = precision[:-1]
    valid_recall = recall[:-1]
    valid_thresholds = thresholds
    f1 = 2 * (valid_precision * valid_recall) / np.clip(valid_precision + valid_recall, 1e-9, None)

    threshold_df = pd.DataFrame(
        {
            "threshold": valid_thresholds,
            "precision": valid_precision,
            "recall": valid_recall,
            "f1": f1,
        }
    )
    best_threshold = float(threshold_df.loc[threshold_df["f1"].idxmax(), "threshold"])
    return threshold_df, best_threshold


def plot_model_comparison(holdout_df: pd.DataFrame, output_path: Path) -> None:
    metric_columns = [
        "test_roc_auc",
        "test_average_precision",
        "test_f1",
        "test_recall",
        "test_precision",
        "test_balanced_accuracy",
    ]
    plot_df = holdout_df.melt(
        id_vars="model",
        value_vars=metric_columns,
        var_name="metric",
        value_name="score",
    )

    plt.figure(figsize=(13, 6))
    sns.barplot(data=plot_df, x="model", y="score", hue="metric", palette="viridis")
    plt.ylim(0.45, 1.02)
    plt.xlabel("")
    plt.ylabel("Score")
    plt.title("Comparacao de metricas no holdout")
    plt.xticks(rotation=20, ha="right")
    plt.legend(title="Metrica", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_roc_and_pr_curves(
    y_test: pd.Series,
    probabilities: dict[str, np.ndarray],
    roc_output_path: Path,
    pr_output_path: Path,
) -> None:
    plt.figure(figsize=(8, 6))
    for name, y_proba in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curvas ROC no holdout")
    plt.legend()
    plt.tight_layout()
    plt.savefig(roc_output_path, dpi=160)
    plt.close()

    plt.figure(figsize=(8, 6))
    for name, y_proba in probabilities.items():
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        plt.plot(recall, precision, linewidth=2, label=f"{name} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curvas Precision-Recall no holdout")
    plt.legend()
    plt.tight_layout()
    plt.savefig(pr_output_path, dpi=160)
    plt.close()


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    output_path: Path,
) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
    disp.plot(cmap="Blues", colorbar=False)
    plt.title(f"Matriz de confusao - {model_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_feature_importance(feature_df: pd.DataFrame, output_path: Path) -> None:
    top_features = feature_df.head(12).iloc[::-1]
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=top_features,
        x="importance",
        y="feature",
        hue="feature",
        palette="crest",
        dodge=False,
        legend=False,
    )
    plt.xlabel("Reducao media em ROC AUC apos permutacao")
    plt.ylabel("")
    plt.title("Importancia por permutacao - melhor modelo")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_feature_set_comparison(comparison_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = comparison_df.melt(
        id_vars="feature_set",
        value_vars=["roc_auc", "average_precision", "f1"],
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(data=plot_df, x="metric", y="score", hue="feature_set", palette="magma")
    plt.ylim(0.75, 1.02)
    plt.xlabel("")
    plt.ylabel("Score medio em cross-validation")
    plt.title("Impacto do feature engineering no XGBoost")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_threshold_tradeoff(threshold_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(threshold_df["threshold"], threshold_df["precision"], label="Precision")
    plt.plot(threshold_df["threshold"], threshold_df["recall"], label="Recall")
    plt.plot(threshold_df["threshold"], threshold_df["f1"], label="F1")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Trade-off entre precision, recall e F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_artifacts(
    cv_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    summary: dict[str, object],
) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    ENGINEERED_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Persist machine-readable outputs so notebook and reports can reuse the same results.
    cv_df.to_csv(ARTIFACTS_DIR / "cv_metrics.csv", index=False)
    holdout_df.to_csv(ARTIFACTS_DIR / "holdout_metrics.csv", index=False)
    feature_df.to_csv(ARTIFACTS_DIR / "permutation_importance.csv", index=False)
    comparison_df.to_csv(ARTIFACTS_DIR / "feature_set_comparison.csv", index=False)
    threshold_df.to_csv(ARTIFACTS_DIR / "threshold_tradeoff.csv", index=False)
    (ARTIFACTS_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def run_pipeline() -> dict[str, object]:
    sns.set_theme(style="whitegrid", palette="viridis")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    ENGINEERED_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Start from the trusted dataset so modeling is decoupled from raw ingestion.
    raw_df = pd.read_parquet(DATASET_PATH)
    engineered_df = engineer_features(raw_df)
    engineered_df.to_parquet(ENGINEERED_DATASET_PATH, index=False)

    X = engineered_df.drop(columns=["churn"])
    y = engineered_df["churn"].astype(int)
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    models = build_models(X.columns.tolist(), scale_pos_weight)

    cv_df = evaluate_with_cross_validation(X, y, models)
    holdout_df, probabilities, X_train, y_train, X_test, y_test = evaluate_on_holdout(X, y, models)

    best_model_name = str(cv_df.loc[0, "model"])
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)
    best_proba = probabilities[best_model_name]
    best_pred = (best_proba >= 0.50).astype(int)

    # Permutation importance is model-agnostic and easier to explain than internal gain metrics.
    permutation = permutation_importance(
        best_model,
        X_test,
        y_test,
        n_repeats=10,
        scoring="roc_auc",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    feature_df = (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance": permutation.importances_mean,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    comparison_df = compare_raw_vs_engineered(raw_df, engineered_df)
    threshold_df, best_threshold = build_threshold_table(y_train, best_model.predict_proba(X_train)[:, 1])

    plot_model_comparison(holdout_df, ARTIFACTS_DIR / "model_metric_comparison.png")
    plot_roc_and_pr_curves(
        y_test,
        probabilities,
        ARTIFACTS_DIR / "roc_curves.png",
        ARTIFACTS_DIR / "precision_recall_curves.png",
    )
    plot_confusion_matrix(
        y_test,
        best_pred,
        best_model_name,
        ARTIFACTS_DIR / "best_model_confusion_matrix.png",
    )
    plot_feature_importance(feature_df, ARTIFACTS_DIR / "best_model_feature_importance.png")
    plot_feature_set_comparison(
        comparison_df,
        ARTIFACTS_DIR / "feature_engineering_impact.png",
    )
    plot_threshold_tradeoff(threshold_df, ARTIFACTS_DIR / "threshold_tradeoff.png")

    summary = {
        "dataset_rows": int(engineered_df.shape[0]),
        "dataset_columns": int(engineered_df.shape[1]),
        "positive_rate": float(y.mean()),
        "best_model": best_model_name,
        "best_model_holdout_metrics": holdout_df.loc[holdout_df["model"] == best_model_name]
        .round(4)
        .to_dict(orient="records")[0],
        "best_threshold_on_train_f1": round(best_threshold, 4),
        "feature_engineering_gain_vs_raw_xgboost": {
            "roc_auc_delta": round(
                float(
                    comparison_df.loc[
                        comparison_df["feature_set"] == "engineered_features", "roc_auc"
                    ].iloc[0]
                    - comparison_df.loc[
                        comparison_df["feature_set"] == "raw_features", "roc_auc"
                    ].iloc[0]
                ),
                4,
            ),
            "average_precision_delta": round(
                float(
                    comparison_df.loc[
                        comparison_df["feature_set"] == "engineered_features",
                        "average_precision",
                    ].iloc[0]
                    - comparison_df.loc[
                        comparison_df["feature_set"] == "raw_features", "average_precision"
                    ].iloc[0]
                ),
                4,
            ),
            "f1_delta": round(
                float(
                    comparison_df.loc[
                        comparison_df["feature_set"] == "engineered_features", "f1"
                    ].iloc[0]
                    - comparison_df.loc[
                        comparison_df["feature_set"] == "raw_features", "f1"
                    ].iloc[0]
                ),
                4,
            ),
        },
        "top_features": feature_df.head(10).round(4).to_dict(orient="records"),
        "feature_notes": FEATURE_NOTES,
    }

    saved_model = save_best_model(
        model_name=best_model_name,
        pipeline=best_model,
        raw_input_columns=raw_df.drop(columns=["churn"]).columns.tolist(),
        feature_columns=X.columns.tolist(),
        target_column="churn",
        threshold=0.50,
        holdout_metrics=summary["best_model_holdout_metrics"],
        project_summary=summary,
    )
    log_model_registration(
        saved_model["metadata"],
        bundle_path=saved_model["bundle_path"],
        metadata_path=saved_model["metadata_path"],
    )
    summary["model_artifacts"] = {
        "bundle_path": str(saved_model["bundle_path"]),
        "metadata_path": str(saved_model["metadata_path"]),
        "model_version": saved_model["metadata"]["model_version"],
    }

    save_artifacts(cv_df, holdout_df, feature_df, comparison_df, threshold_df, summary)
    return summary


def main() -> None:
    summary = run_pipeline()
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
