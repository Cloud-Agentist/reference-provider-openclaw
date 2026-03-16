# Reference Provider: OpenClaw (OpenAI Agents SDK)
#
# Build from reference-provider-openclaw/ directory:
#   docker build -t reference-provider-openclaw .
#
# Run:
#   docker run -p 8082:8082 \
#     -e OPENAI_API_KEY=sk-... \
#     reference-provider-openclaw

FROM python:3.12-slim AS runtime

WORKDIR /app

# Install deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py ./

ENV PORT=8082

RUN groupadd -g 1001 appgroup && useradd -u 1001 -g appgroup appuser
USER appuser

EXPOSE 8082
CMD ["python", "main.py"]
