FROM registry.redhat.io/ubi9/python-39
ADD . .
RUN python -m pip install -r requirements.txt
EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "main.py"]
