import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from environment variables
api_key = os.getenv("GROQ_API_KEY")

# Check if the API key is retrieved successfully
if not api_key:
    raise ValueError("API key is missing in the environment variables.")

# Initialize the Groq client with the API key
client = Groq(api_key=api_key)

# Create the chat completion request
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of fast language models ในภาษาไทย",
        }
    ],
    model="llama-3.3-70b-versatile",
)

# Print the response from the chat completion
print(chat_completion.choices[0].message.content)
