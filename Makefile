.PHONY: setup data train train-gnn serve mlflow test monitor drift-scheduler reproduce stream streaming-demo docker-up docker-down ensemble phase9-eval notebooks

setup:
	pip install -r requirements.txt
	pre-commit install

data:
	python src/data/download_data.py
	python src/data/preprocess.py

train:
	python src/training/train_xgb.py
	python src/training/train_tft.py
	python src/training/train_lgbm.py

# PRD Phase 12 — GNN-GraphSAGE architecture exploration. Kept separate from
# `train:` because the GNN is not part of the deployed ensemble unless step
# 12.2.4's validation-gated integration passes (see docs/adr/ADR-005).
train-gnn:
	python src/training/train_gnn.py

mlflow:
	mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000

serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

stream:
	python src/streaming/producer.py --rate 200 --limit 5000

streaming-demo:
	bash scripts/run_streaming_demo.sh

test:
	pytest tests/ -v --cov=src --cov-report=html

monitor:
	python src/monitoring/drift_reporter.py

drift-scheduler:
	python src/monitoring/drift_scheduler.py

# PRD Phase 8. Export once, then execute the notebook against the cached
# probabilities -- the export loads torch, the notebook must not.
business-impact:
	python scripts/export_test_probabilities.py
	jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 notebooks/05_business_impact.ipynb

# PRD Phase 11 done-when: "all notebooks execute cleanly". 01_eda has no
# model/artifact dependency (runs against data/raw + data/processed only).
# 03_model_comparison and 04_shap_analysis need real trained artifacts in
# models/ (make train) and processed splits (make data) — run those first.
# 05_business_impact is intentionally excluded here: it has its own target
# above because it depends on scripts/export_test_probabilities.py, not on
# nbconvert alone. Fails loudly on the first broken notebook rather than
# continuing past it.
notebooks:
	jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 notebooks/01_eda.ipynb
	jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 notebooks/03_model_comparison.ipynb
	jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 notebooks/04_shap_analysis.ipynb

# PRD Phase 9: after retraining the affected model(s), re-fit the blend +
# operating threshold (writes reports/ only). Add PROMOTE=1 to also freeze it
# into models/ensemble.json — do that only for a step that meets the gate.
ensemble:
	python scripts/run_ensemble_eval.py $(if $(PROMOTE),--promote,)

phase9-eval:
	python scripts/export_test_probabilities.py
	python scripts/run_phase9_eval.py --step "$(STEP)" \
		--derive-threshold-from reports/ensemble_val_probabilities.npz \
		--json reports/phase9_$(STEP).json

reproduce:
	make data && make train

docker-up:
	docker compose up -d

docker-down:
	docker compose down
