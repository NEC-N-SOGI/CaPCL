project_name = capcl
image_name = $(USER)-$(project_name)
container_name = $(USER)-$(project_name)
gpu = all
gpu_option = $(if $(filter all,$(gpu)),all,"device=$(gpu)")
# Optional: options = --mount type=bind,source=/path/to/your/data/,target=/$(project_name)/data/share
options =
command =


run.%: .build.% ## Run a command in a container (% = gpu, ci)
	docker run -it --rm --name=$(container_name) --ipc=host \
		--entrypoint="" \
		--gpus $(gpu_option) \
		-v $$(pwd):/$(project_name) \
		-v /${project_name}/.venv \
		$(options) \
		$(image_name) \
		bash -c "$(command)"

run.ci: .build.ci
	docker run --rm --name=$(container_name) --ipc=host \
		--entrypoint="" \
		-v $$(pwd):/$(project_name) \
		-v /${project_name}/.venv \
		$(options) \
		$(image_name) \
		bash -c "$(command)"

enter.%: up.% ## Enter a container (% = gpu, ci)
	;

up.%: .build.% ## Launch a container (% = gpu, ci)
	docker run -it --rm --name=$(container_name) --ipc=host \
		--gpus $(gpu_option) \
		-v $$(pwd):/$(project_name) \
		-v /${project_name}/.venv \
		$(options) \
		$(image_name)

up.ci: .build.ci
	docker run --rm --name=$(container_name) --ipc=host \
		-v $$(pwd):/$(project_name) \
		-v /${project_name}/.venv \
		$(options) \
		$(image_name)

build.%: .build.% ; ## Build an image in the environment (% = gpu, ci)

.PRECIOUS: .build.%
.build.%: environments/%/build_args.sh environments/Dockerfile pyproject.toml uv.lock
	. $< ; \
	DOCKER_BUILDKIT=1 docker build -t $(image_name) \
		--progress=plain \
		--target=$${BUILD_TARGET} \
		--build-arg BASE_IMAGE=$${BASE_IMAGE} \
		--build-arg PROJECT_NAME=$(project_name) \
		--build-arg PROJECT_DIRECTORY=/$(project_name) \
		--build-arg PYTHON_VERSION=3.12 \
		--build-arg LOCAL_UID=$$(id -u) \
		--build-arg LOCAL_GID=$$(id -g) \
		--build-arg LOCAL_USER_NAME=$$(id -un) \
		-f environments/Dockerfile \
		.
	touch $@

venv: uv.lock pyproject.toml ## Create venv and install the project
	uv sync --all-extras --system-certs

.PRECIOUS: uv.lock
uv.lock:
	uv lock --system-certs

.PHONY: ruff-format
ruff-format:
	uv run --only-group workflow ruff check --fix src
	uv run --only-group workflow ruff format src

.PHONY: ruff-format-check
ruff-format-check:
	uv run --only-group workflow ruff format --check src

.PHONY: ruff-lint
ruff-lint:
	uv run --only-group workflow ruff check src --exclude dev.py --exclude **/org_module/

.PHONY: mdformat
mdformat:
	uv run --only-group workflow mdformat *.md

.PHONY: mdformat-check
mdformat-check:
	uv run --only-group workflow mdformat --check *.md

.PHONY: mypy
mypy:
	uv run --only-group workflow mypy --install-types --non-interactive src

.PHONY: pyright
pyright:
	uv run --only-group workflow pyright src

.PHONY: ty
ty:
	uv run --only-group workflow ty check src

.PHONY: analyze
analyze:
	uvx pyscn analyze src

.PHONY: complexipy
complexipy:
	uvx complexipy src

.PHONY: validate-project
validate-project:
	uv run --only-group workflow validate-pyproject pyproject.toml

.PHONY: precommit
precommit: ## Run pre-commit hooks
	uv run pre-commit run --all-files

.PHONY: prepush
prepush: ## Run pre-push hooks
	uv run pre-commit run --all-files --hook-stage push

.PHONY: format
format: ## Apply formatters to the project
	$(MAKE) ruff-format
	$(MAKE) mdformat

.PHONY: lint
lint: ## Apply formatter checks and linters to the project
	$(MAKE) ruff-format-check
	$(MAKE) ruff-lint
	$(MAKE) ty
	$(MAKE) mdformat-check
	$(MAKE) mypy
	$(MAKE) pyright
	$(MAKE) validate-project

.PHONY: precommit-lint
precommit-lint: ## Apply formatter checks and linters to the project
	$(MAKE) ruff-format-check
	$(MAKE) ruff-lint
	$(MAKE) ty
	$(MAKE) mdformat-check
	$(MAKE) mypy
	$(MAKE) pyright
	$(MAKE) validate-project

.PHONY: sync
sync: ## Sync the project dependencies to the lock file
	uv sync --all-extras --system-certs --no-dev
	uv sync --dev --system-certs
	uv sync --group workflow --system-certs


.PHONY: help
.DEFAULT_GOAL := help

help:
	@grep -E '^[a-zA-Z%._-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
