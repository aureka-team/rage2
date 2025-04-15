include .env

.PHONY: core-build devcontainer-build api-build


core-build:
	[ -e .secrets/.env ] || touch .secrets/.env
	docker compose build rage-core

core-run: core-build
	docker compose run --rm rage-core


devcontainer-build: core-build
	docker compose -f .devcontainer/docker-compose.yml build rage-devcontainer


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

qdrant-flush:
	sudo rm -r ./resources/db/qdrant
	$(info *** WARNING you are deleting all data from qdrant ***)

qdrant-restart: qdrant-stop qdrant-start
