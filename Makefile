.PHONY: install db test app eval
install:
	pip install -e ".[app,dev]"
db:
	pv-agent build-db --synthetic
test:
	pytest -q
app:
	streamlit run app.py
eval:
	pv-agent eval
