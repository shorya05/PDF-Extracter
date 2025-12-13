Simple FastAPI app that fills PDF forms using a RAG-style lookup from a MongoDB knowledge collection.

This project exposes a FastAPI app in `main.py` (the ASGI app object is `app`). The server can be started from Windows Command Prompt (cmd).

## Run in Windows Command Prompt (CMD)

1. (Optional) Create and activate a virtual environment:

	```cmd
	python -m venv .venv
	.venv\\Scripts\\activate
	```

2. Install Python dependencies:

	```cmd
    pip install --upgrade pip
    pip install -r requirements.txt
	```

3. Start the FastAPI server with Uvicorn:

	```cmd
	uvicorn main:app --host 0.0.0.0 --port 8000 --reload
	```

4. Open the interactive API docs in your default browser (from CMD):

	```cmd
	start http://localhost:8000/docs
	```


 2. Run the app

   ```cmd
   uvicorn main:app --host 0.0.0.0 --port 80 --reload
   ```

## Notes and troubleshooting

- Output PDFs are written to the `filled_pdfs` folder (created automatically).
- The app uses a MongoDB Atlas URI and a SentenceTransformer model. If you prefer not to hardcode credentials, update `main.py` to read `MONGO_URI` from an environment variable.
- If `uvicorn` is not found, install it explicitly:

  ```cmd
  pip install uvicorn
  ```

- If you get errors related to `PyMuPDF`/`fitz` or `sentence-transformers`, ensure the packages from `requirements.txt` installed successfully. On Windows, you may need a C++ build toolchain for some binary wheels.

## Quick summary

- Install deps -> `pip install -r requirements.txt`
- Start server -> `uvicorn main:app --host 0.0.0.0 --port 8000 --reload`
- Open docs -> `start http://localhost:8000/docs`

Happy testing!
