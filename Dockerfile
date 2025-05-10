# Step 1: Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy the requirements.txt into the container
COPY requirements.txt /app/

# Step 4: Install dependencies using pip
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the project files into the container
COPY . /app/

# Step 6: Expose port 8000 for the FastAPI app
EXPOSE 8000

# Step 7: Command to run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
