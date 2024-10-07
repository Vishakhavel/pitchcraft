from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

from pitch import check

app = FastAPI()

origins = [
    "http://localhost:3000",  # Allow your frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Directory to save uploaded files
UPLOAD_DIRECTORY = "uploads"

# Create the uploads directory if it doesn't exist
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), goal: str = Form(...), 
    domain: str = Form(...), 
    target_audience: str = Form(...)
    ):
    try:
        print('inside the post function!!!', file, goal, domain, target_audience)

        # Define the file path
        file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)

        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # create an object
        result = check(file, goal, domain, target_audience, file_path)

        # Return success response
        return JSONResponse(content={"message": "File uploaded successfully", "filename": file.filename, "result": result})

    except Exception as e:
        return JSONResponse(content={"message": "Error uploading file", "error": str(e)}, status_code=500)

# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
