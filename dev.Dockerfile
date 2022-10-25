FROM kernai/refinery-parent-images:v0.0.3-common

WORKDIR /app

VOLUME ["/app"]

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY / .

# to run with local version of weak-nlp, clone the weak-nlp repo inside and uncomment
# RUN pip3 install -e weak-nlp

CMD [ "/usr/local/bin/uvicorn", "--host", "0.0.0.0", "--port", "80", "app:app", "--reload" ]
