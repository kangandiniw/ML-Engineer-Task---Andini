from transformers import pipeline

# Load fine-tuned model
pipe = pipeline("text-generation", model="./checkpoints", tokenizer="mistralai/Mistral-7B-Instruct-v0.1")

# Example user intent
def generate_instruction(user_prompt: str):
    formatted_prompt = f"### Instruction:\n{user_prompt}\n\n### Response:\n"
    output = pipe(formatted_prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
    return output[0]["generated_text"]

# Example usage
instruction = generate_instruction("How do I reset my password on the Bukalapak app?")
print(instruction)
