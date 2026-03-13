---
name: ml-pipeline-workflow
description: Build end-to-end ML pipelines from data preparation through model training, validation, deployment planning, and portfolio-ready reporting.
---

# ML Pipeline Workflow

Complete end-to-end ML pipeline orchestration from data preparation through model validation, delivery, and deployment planning.

## Overview

This skill is for building robust ML workflows that go beyond "train a model". It should guide the assistant from data understanding to feature engineering, benchmarking, evaluation, artifact generation, and decision-oriented reporting.

Use this skill when the user needs a professional ML workflow that is reproducible, technically sound, and useful for both engineers and business stakeholders.

## When to Use This Skill

- Building new ML pipelines from scratch
- Extending an existing EDA into feature engineering and predictive modeling
- Designing workflow orchestration for ML systems
- Implementing reproducible training workflows
- Benchmarking multiple models before selecting a winner
- Creating portfolio-ready ML deliverables with notebooks, metrics, and plots
- Planning deployment or productionization after model validation

## What This Skill Provides

### Core Capabilities

1. **Pipeline Architecture**
   - End-to-end workflow design
   - Stage boundaries and dependencies
   - Error handling and retry strategy
   - Reproducibility and artifact organization

2. **Data Preparation**
   - Data validation and quality checks
   - Feature engineering pipelines
   - Data versioning and lineage
   - Train/validation/test split strategy
   - Translating EDA findings into justified derived features

3. **Model Training**
   - Baseline and benchmark model selection
   - Hyperparameter management
   - Experiment tracking hooks
   - Notebook-first training implementation for interactive investigation
   - Programmatic notebook generation with `nbformat` when useful

4. **Model Validation**
   - Cross-validation and holdout evaluation
   - Model comparison workflows
   - Threshold analysis
   - Class-imbalance-aware metric selection
   - Performance regression checks

5. **Delivery and Deployment Readiness**
   - Model artifact packaging
   - Metrics and plot export
   - Notebook-centered workflow
   - Deployment planning and monitoring considerations
   - Plot-by-plot interpretation for stakeholder communication
   - Executive summary for stakeholder-facing delivery

## Default Working Style

When this skill is active, prefer the following behavior:

1. Read the existing EDA or cleaned dataset before proposing modeling.
2. Extract explicit modeling hypotheses from the observed data patterns.
3. Create only feature engineering steps that have statistical or business rationale.
4. Benchmark multiple models before recommending one.
5. Use metrics that fit the problem, especially under class imbalance.
6. Build the workflow directly in the notebook when the deliverable is "Modelagem, Feature Engineering e Avaliacao".
7. Produce machine-readable artifacts and visual outputs.
8. Explain each major chart in plain language: what it shows, why it matters, and what insight it supports.
9. Write result narratives and insights only after running the implemented code, grounding the text in the actual metrics, plots, and artifacts that were produced.
10. End with technical findings plus business insights.
11. Before considering the notebook finished, do a final coherence review to confirm that metrics, charts, conclusions, and executive summary all match the executed results.
12. Add an executive summary after the technical conclusion when the output is meant for presentation, portfolio, or stakeholder review.

## Recommended Workflow

### 1. Context Review

- Read the prior notebook, report, or cleaned dataset.
- Summarize what is already known.
- Identify the target, leakage risks, imbalance, and ready-to-use features.

### 2. Feature Engineering Strategy

- Derive features from concrete EDA findings.
- Prefer rates, ratios, interactions, tenure-normalized signals, behavioral intensity, operational friction, or monetization efficiency when supported by the data.
- Explain why each new feature could improve prediction.
- Avoid arbitrary transformations without rationale.

### 3. Reproducible Notebook Pipeline

- Save notebook deliverables under `notebooks/`.
- Use the exact name `notebooks/02_modeling_evaluation.ipynb` only when the user requests the specific workflow "Modelagem, Feature Engineering e Avaliacao".
- For any other notebook deliverable, choose a notebook name that matches the scope and purpose of the work instead of reusing `02_modeling_evaluation.ipynb`.
- Before writing notebook code, create an implementation plan with the stages, artifacts, analyses, and validation steps that will be added.
- Keep preprocessing tied to training to avoid leakage.
- Use `Pipeline` and `ColumnTransformer` when appropriate.
- Save engineered datasets or downstream artifacts only when they support reuse or reporting.
- Prefer a single self-contained notebook over splitting logic across `src/`, `tmp/`, and helper builders for this specific workflow.
- If manual `.ipynb` editing becomes brittle, generate the notebook programmatically with `nbformat`, which is the official IPython/Jupyter library for reading, writing, and manipulating `.ipynb` files.
- `nbformat` can be used directly from the implementation flow.
- The implementation artifact for this workflow should be the notebook itself, not an intermediate script.

### 4. Model Benchmark

For tabular supervised classification, default to a benchmark set such as:

- Logistic Regression
- Random Forest
- Extra Trees
- Gradient Boosting
- XGBoost
- LightGBM

If some libraries are unavailable, say so and substitute with comparable models.

### 5. Evaluation

For imbalanced classification, default to:

- ROC AUC
- Average Precision
- Recall
- Precision
- F1
- Balanced Accuracy

Preferred validation pattern:

1. Stratified cross-validation
2. Final holdout evaluation
3. Threshold trade-off analysis when operational decisions depend on recall versus precision

### 6. Explainability and Comparison

- Compare raw versus engineered feature sets when feature engineering is a central part of the work.
- Generate feature importance or permutation importance for the selected model.
- Show confusion matrix for the chosen operating threshold.
- Include ROC and Precision-Recall curves.
- Add a short written interpretation for each major chart:
  - what the chart represents,
  - why it matters for the problem,
  - what technical or business insight should be taken from it.
- Base every written interpretation on the executed output itself, citing concrete values, ranking positions, threshold behavior, or feature relevance whenever those results are available.

### 7. Deliverables

Produce as many of these as the project supports:

- Implementation plan
- Analysis notebook in `/notebooks`
- Metrics CSV files
- JSON summary
- Artifact plots
- Engineered dataset
- Business-facing insight summary
- Notebook cells that let the user rerun or inspect the core pipeline logic
- Executive summary section with a concise stakeholder narrative and elegant summary tables
- Notebook structure based on the template in `assets/modeling_feature_engineering_evaluation_notebook_template.md`
- Result text that is explicitly tied to executed outputs rather than hypothetical expectations
- Final notebook validation pass confirming coherence between code outputs, visuals, narrative, and recommendation

## Best Practices

### Pipeline Design

- Modularity: each stage should be independently testable
- Idempotency: rerunning should be safe
- Observability: persist metrics and relevant summaries
- Versioning: track code, data, and outputs
- Failure handling: make debugging straightforward

### Feature Engineering

- Start from evidence in the EDA
- Prefer interpretable transformations when possible
- Quantify whether engineered features improved model quality
- Document the rationale for each important derived feature
- When presenting derived features in the notebook, prefer a professional table with the columns `Feature`, `Categoria`, `O que mede`, `Regra de negocio`, and `Hipotese de impacto no churn`

### Evaluation and Metrics

- Do not rely on accuracy alone in imbalanced problems
- Use stratified splits for classification
- Report both mean and variability in cross-validation when possible
- Compare models side by side in tables and plots
- Keep threshold choice explicit

### Business Translation

- Identify which patterns indicate risk, value, or opportunity
- Separate technical performance from operational usefulness
- End with recommended next actions, not only model scores

## Common Patterns

### EDA to Modeling Pattern

Use this when the project already has an exploratory notebook:

1. Read prior EDA notebook
2. Summarize strongest signals
3. Write an implementation plan
4. Propose feature engineering hypotheses
5. Implement the full workflow in `notebooks/02_modeling_evaluation.ipynb`
6. Train at least 5 models if feasible
7. Evaluate with cross-validation and holdout
8. Generate plots and structured artifacts
9. Add chart-by-chart markdown explanations
10. Use `nbformat` if programmatic notebook assembly is the safest way to build the deliverable
11. Write technical and business insights based on executed outputs
12. Validate the full notebook for consistency across results, charts, and narrative
13. Close with an executive summary

### Portfolio-Ready Modeling Pattern

Use this when the user wants a polished deliverable:

1. Prior EDA recap
2. Implementation plan
3. Feature engineering rationale
4. Notebook implementation in `notebooks/` with a name aligned to the deliverable; use `02_modeling_evaluation.ipynb` only for the specific modeling, feature engineering, and evaluation workflow
5. Benchmark table
6. Visual diagnostics
7. Plot explanations and insight extraction
8. Best-model recommendation
9. Final coherence review of the notebook
10. Executive summary
11. Insights and next steps

### Batch Training Pipeline

```yaml
stages:
  - name: data_preparation
    dependencies: []
  - name: feature_engineering
    dependencies: [data_preparation]
  - name: model_training
    dependencies: [feature_engineering]
  - name: model_evaluation
    dependencies: [model_training]
  - name: reporting
    dependencies: [model_evaluation]
```

## Troubleshooting

### Common Issues

- Pipeline failures: check data availability and path assumptions
- Training instability: review hyperparameters, leakage risk, and data quality
- Poor recall or precision: revisit thresholding and imbalance handling
- No gain from feature engineering: verify whether features are redundant or noisy
- Evaluation mismatch: ensure metrics fit the business objective

### Debugging Steps

1. Validate the dataset entering each stage
2. Check for leakage in preprocessing and splits
3. Re-run a minimal benchmark with fewer models if needed
4. Inspect holdout versus cross-validation gaps
5. Review feature importance and class distribution

## Strong Prompt Template

Use or adapt this prompt when the user wants premium professional ML execution:

```text
Act as a senior machine learning scientist with strong product sense, business reasoning, and production-oriented engineering standards.

Project context:
- I have an existing EDA and/or data-cleaning notebook.
- You must treat that work as required context before proposing feature engineering or modeling.
- The objective is not just to train a model, but to create a robust, reproducible, and decision-useful ML analysis.

Your task:
1. Read and summarize what the previous EDA already established.
2. Extract modeling hypotheses directly from the observed data patterns.
3. Propose and implement feature engineering only when it has clear statistical or business rationale.
4. Start by creating an implementation plan for the modeling workflow.
5. Build a reproducible Python pipeline for preprocessing, training, evaluation, and artifact generation directly in a notebook under `notebooks/`.
6. Train and compare at least 5 relevant models for the problem.
7. Use metrics appropriate for the task, especially for imbalanced classification when applicable.
8. Evaluate models with both cross-validation and holdout testing.
9. Compare the best model on raw features versus engineered features to quantify the value of feature engineering.
10. Produce clear visual diagnostics, including model comparison charts, ROC curves, Precision-Recall curves, confusion matrix, feature importance, and threshold trade-off analysis.
11. Save outputs in structured artifacts such as CSV, JSON, and image files.
12. Deliver both technical findings and business insights, including which variables matter most, what patterns indicate risk or opportunity, and what actions a business team should consider.
13. If programmatic generation is preferable, use `nbformat`, the official IPython/Jupyter library for reading, writing, and manipulating `.ipynb` files, without requiring a fixed helper script path.
14. Make the notebook follow a segmented, well-commented, well-explained structure similar to the template in `assets/modeling_feature_engineering_evaluation_notebook_template.md`.
15. For every major chart, add a short explanation of what it shows, why it matters, and the main insight to take away.
16. Write every result explanation from the executed outputs, using the real values, comparisons, and rankings produced by the implementation.
17. Before finalizing, validate the notebook end to end to ensure that tables, charts, interpretations, conclusion, and executive summary are all coherent with the executed code.
18. After the technical conclusion, add an executive summary with stakeholder-ready narrative and summary tables.

Quality bar:
- Be rigorous, explicit, and pragmatic.
- Do not invent conclusions not supported by the data.
- Do not stop at proposing ideas; implement the full workflow when possible.
- Prefer clean, modular, reusable code.
- Make the final output good enough for a professional portfolio or stakeholder review.

Expected output structure:
- Brief recap of prior EDA findings
- Implementation plan
- Feature engineering strategy with rationale
- Full notebook pipeline implementation in `notebooks/`, using `02_modeling_evaluation.ipynb` only for the specific modeling, feature engineering, and evaluation workflow
- Benchmark results with metrics and plots
- Plot-by-plot explanations and insights
- Final recommendation of the best model
- Executive summary with elegant summary tables
- Business insights and next-step recommendations
```

## Skill Upgrade Guidance

This skill should push the assistant toward a higher bar when the project already has data and exploratory work:

- Do not jump straight into training before reviewing prior analysis
- Treat feature engineering as hypothesis-driven, not decorative
- Default to multi-model benchmarking instead of early model lock-in
- Produce outputs that can be reused in notebooks and reports
- Make notebooks investigative, not only presentational
- Treat chart interpretation as part of the deliverable, not optional polish
- Derive insights from executed outputs, not from hypothetical or generic expectations
- For "Modelagem, Feature Engineering e Avaliacao", require a detailed implementation plan before any notebook code is produced
- Prefer implementing notebook-based workflows directly under `notebooks/`
- Reserve `notebooks/02_modeling_evaluation.ipynb` for the specific workflow "Modelagem, Feature Engineering e Avaliacao"
- If notebook assembly is too large for safe manual editing, use `nbformat` directly to generate the notebook structure
- If the notebook is intended to be self-contained, do not rely on `from src import ...` inside the notebook itself
- Follow the notebook template in `assets/modeling_feature_engineering_evaluation_notebook_template.md`
- For feature engineering documentation, standardize the notebook presentation around the table columns `Feature`, `Categoria`, `O que mede`, `Regra de negocio`, and `Hipotese de impacto no churn`
- Reserve a final executive section for stakeholders after the technical close
- Explain the result in a way that serves both technical review and business discussion
- Add a final notebook QA pass that checks whether the written story still matches the executed outputs

## Related Skills

- `python-patterns`: for Python style, structure, and maintainability
- `experiment-tracking-setup`: for MLflow or W&B integration
- `hyperparameter-tuning`: for deeper optimization
- `model-deployment-patterns`: for production deployment strategy

## Examples

- See `examples/how_to_activate_modeling_feature_engineering_evaluation.md` for a strong example of how to invoke this skill and request a professional end-to-end modeling workflow.
