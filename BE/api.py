from dotenv import load_dotenv
import os

# Specify the path to the .env file
dotenv_path = os.path.join("G:\\Pycharm_directory\\TRANSACTION_RAG\\.venv\\.env")  # Update this to the actual path
load_dotenv(dotenv_path=dotenv_path)

# Load the API key
api_key = os.getenv("OPENAI_API_KEY")

print(api_key)