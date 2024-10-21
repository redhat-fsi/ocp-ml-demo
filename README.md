python -m pip install -r requriements.txt

streamlit run main.py

Use Arize Phoneix instead of LangSmith to trace the path

But if you do not need tracing set INSTRUMENT_TRACE to False

To run Arize Phoneix For Docker/Podman

podman/docker pull arizephoenix/phoenix:latest
podman/docker run -p 6006:6006 arizephoenix/phoenix:latest

Trace it at
http://localhost:6006/

To handle images we need tesseract for Ubuntu install it using following command

sudo apt install tesseract-ocr

For Chroma

To handle error while building new chroma package
sudo apt-get install build-essential
sudo apt-get install python3-dev
sudo apt-get install libssl-dev
sudo apt-get install g++

For PGVector

podman run --name pgvector-container -e POSTGRES_USER=rhnasa -e POSTGRES_PASSWORD=rhnasa123 -e POSTGRES_DB=rhnasademo -p 6024:5432 -d docker.io/pgvector/pgvector:pg17
