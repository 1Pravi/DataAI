import google.generativeai as genai

genai.configure(api_key="AIzaSyAs6n5qFoe987XofhIuNzzQzzdAy6hDPUE")

models = genai.list_models()

for m in models:
    print(m.name)
