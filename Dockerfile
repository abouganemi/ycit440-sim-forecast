# Training image. The PyTorch ROCm wheels bundle the ROCm user-space
# libraries, so a plain Python base works; only the host kernel driver
# (amdgpu) and the /dev/kfd + /dev/dri devices are needed at run time.
FROM ghcr.io/astral-sh/uv:0.12.19-python3.14-trixie-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first so code changes don't invalidate the (multi-GB) torch layer.
# --no-dev drops only the dev group; the gpu group is a default group.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project

COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH"

# Non-root, uid 1000 so files written to the mounted models/ belong to the
# host user. GPU access comes from --group-add video/render (Docker) or
# supplementalGroups (k3s) at run time. USER is numeric so Kubernetes can
# verify runAsNonRoot; useradd still runs so HOME exists for `docker run`.
RUN useradd --create-home --uid 1000 app
USER 1000:1000

# data/ and models/ are mounted (Docker) or come from PVCs (k3s), never
# baked in (see .dockerignore).
CMD ["python", "scripts/check_gpu.py"]
