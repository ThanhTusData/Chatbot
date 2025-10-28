.PHONY: install test lint format clean train serve build-kb docker-build docker-up

install:
\tpip install -r requirements.txt
\tpython -m spacy download en_core_web_sm

install-dev:
\tpip install -r requirements-dev.txt
\tpre-commit install

test:
\tpytest tests/ -v --cov=src --cov-report=html

test-unit:
\tpytest tests/unit/ -v

test-integration:
\tpytest tests/integration/ -v

lint:
\tflake8 src/ tests/
\tmypy src/

format:
\tblack src/ tests/
\tisort src/ tests/

clean:
\tfind . -type d -name __pycache__ -exec rm -rf {} +
\tfind . -type f -name '*.pyc' -delete
\trm -rf .pytest_cache .coverage htmlcov

train:
\tpython src/training/train_intent.py --data data/processed/training_data.json --output models/intent

build-kb:
\tpython scripts/build_kb_index.py --kb-dir data/kb --output indexes/kb

serve-api:
\tpython src/serving/fastapi_app.py

serve-web:
\tpython src/web/flask_app.py

docker-build:
\tdocker build -f deployment/docker/Dockerfile -t chatbot-ml-nlp:latest .

docker-up:
\tdocker-compose -f deployment/compose/docker-compose.yaml up -d

docker-down:
\tdocker-compose -f deployment/compose/docker-compose.yaml down

k8s-deploy:
\tkubectl apply -f deployment/kubernetes/

k8s-delete:
\tkubectl delete -f deployment/kubernetes/