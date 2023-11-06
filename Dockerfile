# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.10-slim-bullseye

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

WORKDIR /app
EXPOSE 80
COPY . /app

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt



#HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
#ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port", "8501"]

CMD ["streamlit", "run", "streamlit_app.py"]
