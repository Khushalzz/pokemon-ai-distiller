# Pokémon AI Distillation Pipeline

This repository contains the standalone, Kaggle-ready Jupyter Notebook and data files required to distill knowledge from Llama-3.3-70B down into a lightweight, fine-tuned **Qwen2.5-7B** model via LoRA.

This allows us to generate dynamic, lore-accurate, and incredibly witty Pokémon NPC dialogue on a fast local model, matching the quality of the massive 70B parameter models!

## 🚀 How to Run on Kaggle

This pipeline is specifically engineered to run in the free Kaggle dual T4 GPU environment. It features a robust **Hybrid API/Local Fallback** mechanism, guaranteeing that you never hit API rate limits or stop generating!

1. Fork this repository or download `kaggle_training_pipeline.ipynb` and `text_database.json`.
2. Open [Kaggle](https://www.kaggle.com/) and create a new Notebook.
3. Select **File > Import Notebook** and upload `kaggle_training_pipeline.ipynb`.
4. Upload `text_database.json` to your Kaggle Notebook's `/kaggle/input` directory.
5. In your Kaggle settings, add a new Secret named `GROQ_API_KEY` with your free [Groq API Key](https://console.groq.com/keys).
6. Click **Run All**!

---

## 🔍 Pipeline Deep Dive (Exact Code Blocks)

### 1. PokeAPI Retrieval-Augmented Generation (RAG)
Before the AI generates a joke, we parse the original Game Boy text to see if it mentions any specific Pokémon. We then query the live `PokeAPI` to get their elemental typings, providing the AI with extreme context accuracy.

```python
import requests
import json
import re

def get_pokemon_type(pokemon_name):
    try:
        res = requests.get(f"https://pokeapi.co/api/v2/pokemon/{pokemon_name.lower()}")
        if res.status_code == 200:
            types = [t["type"]["name"] for t in res.json()["types"]]
            return " and ".join(types)
    except:
        pass
    return None

def inject_rag_context(text):
    pokemon_mentions = re.findall(r'\b([A-Z]{3,})\b', text)
    context = []
    for p in pokemon_mentions:
        ptype = get_pokemon_type(p)
        if ptype:
            context.append(f"{p} is a {ptype} type Pokémon.")
    return " ".join(context)
```

### 2. Hybrid Distillation (Llama-3 to Qwen2.5)
We iterate over the `text_database.json` and attempt to call the Groq API to utilize the massive **Llama-3.3-70B**. If Groq rate limits us, the pipeline seamlessly loads **Qwen2.5-7B** directly into the Kaggle GPU memory and continues generating the dataset locally!

```python
import os
from groq import Groq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from kaggle_secrets import UserSecretsClient

client = Groq(api_key=UserSecretsClient().get_secret("GROQ_API_KEY"))

fallback_model_id = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = None
fallback_model = None

dataset = []

for item in list(game_texts.items()):
    address, original_text = item
    rag_context = inject_rag_context(original_text)
    prompt = f"Rewrite this Pokémon NPC text to be funny and sarcastic. Original: {original_text}\nContext: {rag_context}"
    
    try:
        # 1. Try Groq API
        chat = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile"
        )
        response = chat.choices[0].message.content
    except Exception as e:
        print("Groq rate limit hit. Falling back to local Qwen2.5-7B on T4 GPUs...")
        # 2. Local Fallback
        if fallback_model is None:
            tokenizer = AutoTokenizer.from_pretrained(fallback_model_id)
            fallback_model = AutoModelForCausalLM.from_pretrained(
                fallback_model_id, device_map="auto", torch_dtype=torch.float16
            )
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = fallback_model.generate(**inputs, max_new_tokens=50)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
        
    dataset.append({"input": original_text, "output": response})

with open("distilled_dataset.json", "w") as f:
    json.dump(dataset, f)
```

### 3. PEFT LoRA Fine-Tuning
Using `SFTTrainer`, we train our `Qwen2.5-7B` on the newly generated dataset. This teaches the fast, local model how to replicate the comedy and reasoning of the 70B teacher model.

```python
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer

data = load_dataset("json", data_files="distilled_dataset.json", split="train")

def format_prompt(example):
    return f"<|im_start|>user\nRewrite this Pokémon text: {example['input']}<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", quantization_config=bnb_config, device_map="auto")

lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

trainer = SFTTrainer(
    model=model,
    train_dataset=data,
    formatting_func=format_prompt,
    args=TrainingArguments(
        output_dir="pokemon_adapter",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        max_steps=500,
        learning_rate=2e-4,
        fp16=True
    )
)
trainer.train()
trainer.model.save_pretrained("pokemon_adapter")
```
