# Dockerfile for ICRS Sandbox
# Builds a self-contained image that runs the ranking pipeline on a
# 50-candidate sample and produces a ranked CSV.
#
# Build:
#   docker build -t icrs-sandbox .
#
# Run (see ranked results in stdout, CSV written to /out inside container):
#   docker run --rm icrs-sandbox
#
# Run (extract the CSV to your current directory):
#   docker run --rm -v "$PWD:/out" icrs-sandbox
#
# The pipeline uses only the Python standard library — no pip install needed.
# Completes in < 5 seconds on CPU. No internet access required.

FROM python:3.11-slim

LABEL org.opencontainers.image.title="ICRS Sandbox"
LABEL org.opencontainers.image.description="Intelligent Candidate Ranking System - Sandbox"
LABEL org.opencontainers.image.licenses="MIT"

# Create non-root user for security
RUN groupadd -r appgroup && useradd -r -g appgroup -u 1001 appuser

# Create output directory writable by non-root user (for volume mounts)
RUN mkdir -p /out && chown appuser:appgroup /out

WORKDIR /app

# Make /app writable by non-root user (for output files)
RUN chown appuser:appgroup /app

# Copy source code and sandbox sample data
COPY --chown=appuser:appgroup . .

# Switch to non-root user
USER appuser

# Entry point: runs the sandbox runner
# Default: writes CSV to /out/sandbox_submission.csv, prints results to stdout
# Override: docker run ... --out /custom/path.csv
ENTRYPOINT ["python", "sandbox/sandbox.py"]
CMD ["--out", "/out/sandbox_submission.csv"]
