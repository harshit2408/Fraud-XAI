.PHONY: setup data train serve mlflow test monitor reproduce stream docker-up docker-down

setup:
	pip install -r requirements.txt
	pre-commit install

data:
	python src/data/download_data.py
	python src/data/preprocess.py

train:
	python src/training/train_xgb.py
	python src/training/train_tft.py

mlflow:
	mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000

serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

stream:
	python src/streaming/producer.py --rate 200 --limit 5000

test:
	pytest tests/ -v --cov=src --cov-report=html

monitor:
	python src/monitoring/drift_reporter.py

reproduce:
	make data && make train

docker-up:
	docker compose up -d

docker-down:
	docker compose down
