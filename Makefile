.PHONY: core-build core-run devcontainer-build api-build retriever-run


core-build:
	docker compose build rage-core

core-run: core-build
	docker compose run --rm rage-core


devcontainer-build: core-build
	docker compose build rage-devcontainer


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


test-retriever: devcontainer-build
	docker compose run --rm -e PYTHONPATH=/workspace/src --entrypoint="python -m rage.scripts.qdrant.run_retriever" rage-devcontainer
