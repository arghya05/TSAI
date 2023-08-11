# Stage 1: Builder/Compiler
FROM python:3.7-slim-buster AS build

COPY requirements.txt .

# Create the virtual environment.
RUN python3 -m venv /venv
ENV PATH=/venv/bin:$PATH

RUN pip3 install --no-cache-dir -U pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.7-slim-buster

COPY --from=build /venv /venv
ENV PATH=/venv/bin:$PATH
ENV GRADIO_SERVER_PORT 80

WORKDIR /code

COPY vision.py .

EXPOSE 80
ENTRYPOINT ["python3", "vision.py"]
