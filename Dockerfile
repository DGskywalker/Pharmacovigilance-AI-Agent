FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY app.py ./
COPY evals ./evals
RUN pip install --no-cache-dir -e ".[app]"
RUN pv-agent build-db --synthetic
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
