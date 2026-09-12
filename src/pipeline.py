import requests

response = requests.post(
    url="http://localhost:11434/api/chat",
    json={
        "model": "qwen2.5:7b",
        "messages": [
            {"role": "system", "content": "אתה עוזר שעונה בעברית בלבד."},
            {"role": "user", "content": "מהי עיר הבירה של מיקרונזיה?"},
        ],
        "stream": False,
    },
)

print(response.json()["message"]["content"])
