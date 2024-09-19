FROM python:3.12-slim

RUN pip install poetry

WORKDIR /app

COPY ./pyproject.toml ./poetry.lock /app/

RUN poetry install

COPY . /app/

CMD ["poetry", "run" , "python",  "chatmodels\basic-chatmodels.py"]