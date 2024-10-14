FROM python:3.10

WORKDIR /app

# Copy requirements first to leverage Docker caching
COPY ./app/requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Copy the rest of your app code
COPY ./app /app

EXPOSE 8000

# CMD ["fastapi", "dev", "app.py"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
