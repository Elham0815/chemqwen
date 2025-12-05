import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import json

# Load the chemistry dataset first
with open('chemistry_dataset.json', 'r') as f:
    dataset = json.load(f)

# Load model and tokenizer FIRST - this was the error!
print("Loading model and tokenizer...")
model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Set up padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# NOW format the data with the tokenizer available
formatted_data = []
for item in dataset:
    text = f"### Instruction: {item['instruction']}\n\n### Input: {item['input']}\n\n### Response: {item['output']}{tokenizer.eos_token}"
    formatted_data.append({"text": text})

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.float32  # Fixed: use dtype instead of torch_dtype
)

# Tokenization function with labels
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )
    # For causal LM, labels = input_ids
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

# Create dataset
raw_dataset = Dataset.from_list(formatted_data)

# Tokenize dataset
print("Tokenizing dataset...")
tokenized_dataset = raw_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./chemistry_model_final",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    learning_rate=5e-5,
    logging_steps=1,
    save_strategy="epoch",
    report_to="none",
    fp16=False,
    remove_unused_columns=False,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Start training
print("🌟 Starting training - this time it WILL work! 🌟")
trainer.train()

# Save the model
print("✅ Training complete! Saving model...")
model.save_pretrained("./chemistry_model_final")
tokenizer.save_pretrained("./chemistry_model_final")
print("🎉 Success! Your chemistry model is ready!")
