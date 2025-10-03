import requests
import os

url = "http://localhost:3000/ask"
pdf_path = "../data/computer-history.pdf"

if not os.path.exists(pdf_path):
    print(f"Error: PDF file not found at {pdf_path}")
    exit(1)

files = {"pdf_file": ("computer-history.pdf", open(pdf_path, "rb"), "application/pdf")}
data = {
    "question": "Explain microcomputer in detail.",
    "api_key": "xxxx"  # Replace with your actual Google API key
}

try:
    response = requests.post(url, files=files, data=data)
    if response.status_code == 200:
        print(response.json())
    else:
        print(f"Error: {response.status_code}, {response.text}")
except requests.exceptions.ConnectionError:
    print("Connection Error: Ensure the FastAPI server is running on http://localhost:3000")
except Exception as e:
    print(f"Unexpected error: {str(e)}")