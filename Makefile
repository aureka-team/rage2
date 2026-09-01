.PHONY: core-build core-run devcontainer-build api-build api-run api-up api-stop api-restart api-create-zaratustra api-retrieve neighboring-text-chunks retriever-run


core-build:
	docker compose build rage-core

core-run: core-build
	docker compose run --rm rage-core


devcontainer-build: core-build
	docker compose build rage-devcontainer


api-build: core-build
	docker compose build rage-api

api-run: api-build
	docker compose run --rm --service-ports rage-api

api-up: api-build
	docker compose up rage-api -d

api-stop:
	docker compose stop rage-api

api-restart: api-stop api-up


redis-start:
	docker compose up -d rage-redis

redis-stop:
	docker compose stop rage-redis

redis-flush:
	docker compose exec rage-redis redis-cli FLUSHALL

redis-restart: redis-stop redis-start


qdrant-start:
	docker compose up -d rage-qdrant

qdrant-stop:
	docker compose stop rage-qdrant

qdrant-flush: qdrant-stop
	sudo rm -r ./resources/db/qdrant
	$(info *** WARNING you are deleting all data from qdrant ***)
	docker compose up -d rage-qdrant

qdrant-restart: qdrant-stop qdrant-start


api-create-collection:
	docker compose exec -e PYTHONPATH=/workspace/src -e RAGE_API_URL=http://rage-api:$${API_PORT:-8000} rage-devcontainer python -m rage.scripts.api.create_collection

api-retrieve:
	docker compose exec -e PYTHONPATH=/workspace/src -e RAGE_API_URL=http://rage-api:$${API_PORT:-8000} rage-devcontainer python -m rage.scripts.api.retrieve

neighboring-text-chunks:
	docker compose exec -e PYTHONPATH=/workspace/src rage-devcontainer python -m rage.scripts.qdrant.get_neighboring_text_chunks

test-retriever: devcontainer-build
	docker compose run --rm -e PYTHONPATH=/workspace/src --entrypoint="python -m rage.scripts.qdrant.run_retriever" rage-devcontainer
