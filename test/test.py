import os
from transformers import pipeline

image_path=os.path.join("dataset", "images", "8800279679073.png")

pipe=pipeline(
    "image-text-to-text",
    model="prithivMLmods/Camel-Doc-OCR-080125",
)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": image_path},
            {"type": "text", "text": "Extract raw text."}
        ]
    },
]
result=pipe(text=messages)
print(result[0]["generated_text"])