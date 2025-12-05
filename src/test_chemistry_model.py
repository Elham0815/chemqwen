from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("🚀 Loading your trained chemistry model...")
model_path = "./chemistry_model_final"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)

# Ask a chemistry question
question = "### Instruction: Explain how DFT calculations work for molecular systems\n\n### Input: \n\n### Response:"

inputs = tokenizer(question, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n✨ CHEMISTRY EXPERT RESPONSE:")
print(response.split("### Response:")[-1].strip())
