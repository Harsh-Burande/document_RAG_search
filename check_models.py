import os
from dotenv import load_dotenv
import google.generativeai as genai

# load environment variables
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

# configure gemini
genai.configure(api_key=API_KEY)

# list models
for model in genai.list_models():
    print(model.name, model.supported_generation_methods)


# for model in genai.list_models():
#     if "embedContent" in model.supported_generation_methods:
#         print(model.name)