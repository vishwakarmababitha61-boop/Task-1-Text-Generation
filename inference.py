from transformers import pipeline
import os

generator = pipeline("text-generation", model="distilgpt2")

with open("prompts.txt", "r", encoding="utf-8") as file:
    prompts = file.readlines()

os.makedirs("generated_samples", exist_ok=True)

for i, prompt in enumerate(prompts, start=1):
    prompt = prompt.strip()

    if not prompt:
        continue

    result = generator(
        prompt,
        max_new_tokens=80,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.8
    )

    output = result[0]["generated_text"]

    with open(
        f"generated_samples/prompt_{i:02d}.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write(output)

    print(f"Generated output {i}/20")

print("All outputs generated successfully!")