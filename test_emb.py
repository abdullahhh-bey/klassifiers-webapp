import os
import sys
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('geminiAPI')
if not api_key:
    print('no key found in .env')
    sys.exit(1)

models = ['models/embedding-001', 'models/text-embedding-004', 'models/gemini-embedding-001']
for model in models:
    print(f'Trying {model}...')
    try:
        emb = GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)
        res = emb.embed_query('hello')
        print(f'Success! Dim: {len(res)}')
    except Exception as e:
        print(f'Error: {e}')
