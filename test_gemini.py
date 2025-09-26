import google.generativeai as genai
import os

api_key = os.getenv("GEMINI_API_KEY")
model = os.getenv("GEMINI_MODEL", "gemini-pro")

print(f"Usando modelo: {model}")

genai.configure(api_key=api_key)
try:
    model_client = genai.GenerativeModel(model)
    response = model_client.generate_content("Dame un resumen del mercado crypto actual")
    print("Respuesta Gemini:", response.text)
except Exception as e:
    print("❌ Error al usar Gemini:", e)
