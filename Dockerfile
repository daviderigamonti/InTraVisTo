FROM python:3.12

WORKDIR /app/

COPY requirements.txt .
COPY ./submodules/transformer_wrappers/requirements.txt ./transformer_wrappers_requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install \ 
    -r requirements.txt \
    -r ./transformer_wrappers_requirements.txt

COPY . .

ENV PYTHONPATH "${PYTHONPATH}:./submodules/transformer_wrappers/src"

ENV HF_HOME=./huggingface
ENV TRANSFORMERS_CACHE=./huggingface

ENV HOST 0.0.0.0

EXPOSE 8892

CMD [ "python", "./src/InTraVisTo.py" ]