# Customer Churn Prediction System — Submission Checklist

## Objectives and required components

- [x] Publicly available Customer Churn Dataset source documented and supplied train/test CSVs included.
- [x] Load, validate, clean, and describe datasets; remove one blank training row.
- [x] Exploratory data analysis (summary JSON and six training/test charts).
- [x] Numeric imputation, scaling, categorical imputation, and one-hot encoding within one Pipeline.
- [x] Separate train/validation split; compare two classification algorithms; independent external test evaluation.
- [x] Accuracy, precision, recall, F1, confusion matrix, ROC-AUC, Brier score and report.
- [x] Joblib trained model and JSON model metadata.
- [x] `GET /health`, `POST /predict`, `GET /model/info`, `GET /predictions` with FastAPI.
- [x] Strict Pydantic validation, safe error responses, and logging.
- [x] Responsive HTML/CSS/JavaScript frontend with form, probability, risk level, and insight chart.
- [x] Dockerfile and Compose configuration created.
- [x] SQLite history, Pytest tests, model comparison, permutation feature importance.
- [x] README, screenshots, and GitHub Actions CI workflow.
- [x] Local API, model inference and front-end interaction tested; **17 Pytest tests pass**.

## Still requires the submitter or deployment environment

- [ ] Push the extracted source project to an authenticated GitHub account (instructions in README).
- [ ] Confirm dataset licensing before redistributing in a public repository.
- [ ] Build and test the Docker image where Docker CLI/daemon is available; Docker was unavailable in the build environment.
- [ ] If intended for operational use, investigate the significant data shift and improve model generalization/calibration. See `reports/model_evaluation.md`.

The application is complete as a **functional academic prototype**, not validated for operational retention decisions.
