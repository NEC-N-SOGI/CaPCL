ARG PYTHON_VERSION=3.12
ARG BASE_IMAGE=python:${PYTHON_VERSION}-bullseye

###############################################################################
FROM ${BASE_IMAGE} AS dev-env

ARG BASE_IMAGE
ARG PROJECT_NAME=capcl
ARG PROJECT_DIRECTORY=/${PROJECT_NAME}
ARG LOCAL_UID=1000
ARG LOCAL_GID=1000
ARG LOCAL_USER_NAME=user
ARG PYTHON_VERSION

ENV LOCAL_UID=${LOCAL_UID} \
    LOCAL_GID=${LOCAL_GID} \
    LOCAL_USER_NAME=${LOCAL_USER_NAME}

ENV LC_ALL="C.UTF-8" \
    LANG="C.UTF-8" \
    PYTHONPATH=${PROJECT_DIRECTORY}/src


ENV HF_DATASETS_CACHE=${PROJECT_DIRECTORY}/data/share/temp/ \
    HF_HOME=${PROJECT_DIRECTORY}/data/share/temp/ \
    TORCH_HOME=${PROJECT_DIRECTORY}/data/share/temp/

ENV UV_SYSTEM_PYTHON=0 \
    UV_SYSTEM_CERTS=true

# Set the timezone
ENV TZ=Asia/Tokyo
RUN apt-get update \
    && apt-get install -y tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && echo "${TZ}" > /etc/timezone \
    && rm /etc/localtime \
    && ln -s /usr/share/zoneinfo/Asia/Tokyo /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf ~/.cache

# Install the dependencies
RUN apt update \
    && apt install --no-install-recommends -y \
        curl\
        ca-certificates \
        git \
        make \
        cmake \
        vim \
        gcc \
        g++ \
        default-jre \
        graphviz \
        tmux \
        libpq-dev \
        libgl1-mesa-dev \
        python3-dev \
    && echo "Y" | apt install sudo -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt update \
    && apt install -y zsh bat \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf ~/.cache

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install libraries
RUN apt update \
    && DEBIAN_FRONTEND=noninteractive apt install -y libtiff5-dev libopenjp2-7-dev \
        libfreetype6-dev liblcms2-dev libwebp-dev libharfbuzz-dev libfribidi-dev \
    && DEBIAN_FRONTEND=noninteractive apt install --no-install-recommends -y \
            cmake \
            g++ \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y libjpeg-turbo8-dev zlib1g-dev libfreetype6-dev libfribidi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf ~/.cache

RUN apt update \
    && apt install -y libopenmpi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf ~/.cache

# Configure the local user and group
# username = ${LOCAL_USER_NAME}, uid = ${LOCAL_UID}, gid = ${LOCAL_GID}
RUN existing_user="$(getent passwd "${LOCAL_UID}" | cut -d: -f1)" \
    && if [ -n "${existing_user}" ] && [ "${existing_user}" != "${LOCAL_USER_NAME}" ]; then \
        usermod --login "${LOCAL_USER_NAME}" "${existing_user}"; \
        usermod --home "/home/${LOCAL_USER_NAME}" --move-home "${LOCAL_USER_NAME}"; \
    elif ! id -u "${LOCAL_USER_NAME}" > /dev/null 2>&1; then \
        useradd --uid "${LOCAL_UID}" --create-home "${LOCAL_USER_NAME}"; \
    else \
        usermod --uid "${LOCAL_UID}" "${LOCAL_USER_NAME}"; \
    fi \
    && if ! getent group "${LOCAL_GID}" > /dev/null; then \
        if getent group "${LOCAL_USER_NAME}" > /dev/null; then \
            groupmod --gid "${LOCAL_GID}" "${LOCAL_USER_NAME}"; \
        else \
            groupadd --gid "${LOCAL_GID}" "${LOCAL_USER_NAME}"; \
        fi; \
    fi \
    && usermod --gid "${LOCAL_GID}" "${LOCAL_USER_NAME}"

RUN mkdir -p /etc/sudoers.d \
    && usermod -aG sudo ${LOCAL_USER_NAME} \
    && echo "${LOCAL_USER_NAME} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers.d/trusted-user

USER ${LOCAL_USER_NAME}
WORKDIR ${PROJECT_DIRECTORY}

ENV HOME=/home/${LOCAL_USER_NAME}

# Install the project's dependencies using the lockfile and settings
COPY --chown=${LOCAL_UID}:${LOCAL_GID} pyproject.toml ${PROJECT_DIRECTORY}/pyproject.toml
COPY --chown=${LOCAL_UID}:${LOCAL_GID} uv.lock ${PROJECT_DIRECTORY}/uv.lock

RUN --mount=type=cache,target=/home/${LOCAL_USER_NAME}/.cache/uv,uid=${LOCAL_UID},gid=${LOCAL_GID} \
    uv venv -p ${PYTHON_VERSION} \
    && uv sync --all-extras --system-certs --no-dev --no-install-project \
    && uv sync --dev --system-certs --no-install-project \
    && uv sync --group workflow --system-certs --no-install-project

ENTRYPOINT [ "zsh" ]
CMD [ "--login" ]
