
# training_pipeline.py

from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset

# Step 1: Load model and tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 2: Prepare for QLoRA fine-tuning
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# Step 3: Load and preprocess the dataset
dataset = load_dataset("json", data_files={"train": "data/train.jsonl", "validation": "data/val.jsonl"})

def preprocess(example):
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
    response = "\n".join(example['output']) if isinstance(example['output'], list) else example['output']
    return tokenizer(prompt + response, truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(preprocess)

# Step 4: Set up training configuration
training_args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10
)

# Step 5: Launch Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer
)

# Step 6: Start training
trainer.train()

