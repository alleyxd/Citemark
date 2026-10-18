FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY citemark ./citemark
RUN pip install --no-cache-dir .
COPY examples ./examples
# build the example index at image build time; mount your own documents and rebuild for real use
RUN citemark index examples/handbook --out /app/.citemark
ENV CITEMARK_INDEX=/app/.citemark
EXPOSE 8080
CMD ["uvicorn", "citemark.api:app", "--host", "0.0.0.0", "--port", "8080"]
