# ML-Engineer-Task---Andini


# 1. Open-Source LLM: Mistral-7B Instruct

Mistral-7B Instruct represents a well-balanced open-source LLM that combines strong instruction-following behavior, structured output generation, computational efficiency, and commercial viability. Its capabilities make it highly effective for transforming user intent into structured, task-oriented outputs — both in research contexts and production-level systems.

### 1. Purpose-Built for Instruction Following
Mistral-7B Instruct has been fine-tuned specifically on instruction-following datasets, enabling it to:
•	Accurately interpret user intent expressed in natural language
•	Produce clear, step-by-step, structured outputs
This makes it particularly effective in scenarios that require generating procedural or task-oriented content based on minimal user input.

### 2. High Performance in a Compact Model Size
Despite its relatively small size of 7 billion parameters, Mistral-7B Instruct demonstrates performance comparable to, or in some cases exceeding, larger models such as LLaMA-2 13B — particularly in instruction-following and reasoning tasks. This efficiency makes it well-suited for deployment in environments with limited computational resources.

### 3. Permissive Open-Source License
The model is released under the Apache 2.0 license, which allows:
•	Full commercial use
•	Modification and redistribution
•	Transparent evaluation and auditing
Such licensing is crucial for organizations seeking to build proprietary systems while retaining control over their AI infrastructure.

### 4. Structured Output Capabilities
Mistral-7B Instruct excels at generating structured formats such as:
•	JSON and YAML
•	Function calls and configuration files
•	Ordered procedural instructions
These capabilities are essential for applications involving automation, API orchestration, workflow generation, or integration with downstream systems.

### 5. Strong Ecosystem Support
The model is supported by a growing ecosystem, including:
•	Integration with frameworks such as LangChain and LlamaIndex (for RAG and tool use)
•	Compatibility with inference engines like vLLM, TGI, and Ollama
•	Readily usable through Hugging Face’s Transformers library
This makes it easy to adopt, scale, and integrate into real-world applications.


# 2.	Dataset Design and Preparation: 

### 1. What kind of data?
For fine-tuning the Mistral 7B Instruct model on task-oriented instruction generation, I selected Option 1: Manually created task-based JSONL dataset.

This format aligns well with the model’s instruction-tuning design, where each data point includes a clear instruction field and a structured, step-by-step output.

### 2. How to collect and annotate?
Semi-automatic generation using GPT-4 as a teacher model, followed by human validation and refinement.

Workflow:
1. Curate task-based user questions relevant to domains such as e-commerce, fintech, or education.
2. Use GPT-4 to generate multi-step outputs for each instruction.
3. Manually review and revise the generated outputs to ensure factual accuracy, logical step flow, and consistent formatting.

Pros:
Rapid bootstrapping of a large dataset with minimal manual effort.
Leverages the capabilities of high-performing LLMs for realistic instruction formatting.
Enables scalable creation while retaining control over quality.

Cons:
Generated steps may include hallucinations or incomplete logic.
Still requires human-in-the-loop QA to ensure reliability.
This approach offers an optimal balance between scalability and data quality.

### 3. Preprocessing steps
To prepare the dataset for fine-tuning Mistral 7B Instruct, the following preprocessing steps are implemented:

1. Cleaning
•	Remove duplicate entries.
•	Normalize text (e.g., convert to lowercase, standardize punctuation).
•	Correct grammatical issues or formatting inconsistencies.

2. Tokenization
•	Use MistralTokenizer from the Hugging Face transformers library.
•	Ensure total token length per sample does not exceed the model’s context window (e.g., 4096 tokens).
•	Apply consistent formatting (e.g., bullet points for steps, newline between actions if applicable).

3. Splitting
•	Train set: 80%
•	Validation set: 10%
•	Test set: 10%
•	Performed using stratified or random splitting while preserving domain diversity.

### 4. Handling special cases
1. Self-correction and Edge Cases
To ensure robustness and ethical compliance during dataset preparation, several strategies are implemented for handling edge cases such as data imbalance, sensitive information, and generalization across diverse instruction types.

2. Data Imbalance
Apply class-weighted loss functions during training and selectively oversample underrepresented task types during dataset creation. This improves the model’s ability to learn from rare or low-frequency instruction formats while avoiding overfitting. However, this approach requires careful tuning and monitoring to prevent unintended bias or instability during training.

3. Sensitive Data,
Use regular expressions to automatically detect and remove personally identifiable information (PII), such as emails, phone numbers, or account numbers. While automated filtering is fast and scalable, I also include manual review for critical or edge-case samples to ensure data quality and ethical standards are upheld. This hybrid approach balances efficiency with thoroughness.

4. Promote Diversity and Generalization
Collect instructions from multiple domains such as e-commerce, fintech, education, and productivity apps. This exposes the model to a broad range of user intents and language styles. In addition, I employ prompt augmentation techniques like paraphrasing the same instruction in various ways. This encourages the model to generalize better across linguistic variations and user contexts. While such augmentation adds variability, I carefully validate the quality to avoid inconsistencies in task execution logic.


# 3.	Fine-Tuning Strategy: 

### QLoRA (Quantized Low-Rank Adapter) 
To adapt a pre-trained language model to generate structured, task-oriented instructions from natural language prompts, I propose using instruction tuning with QLoRA (Quantized Low-Rank Adapter) — a Parameter-Efficient Fine-Tuning (PEFT) method. This approach enables the model to specialize on instruction-following tasks while preserving its general language understanding capabilities. QLoRA is particularly well-suited for this task due to its computational efficiency, enabling fine-tuning of large models (e.g., Mistral-7B or similar) on consumer-grade hardware with limited VRAM (e.g., 24GB). 

### Key Hyperparameters
To ensure stable and effective fine-tuning, the following hyperparameters will be carefully tuned:
•	Learning Rate (2e-5): QLoRA typically requires a lower learning rate to stabilize updates due to the use of adapters and quantization.
•	Batch Size (8 per device): Set based on GPU memory constraints; combined with gradient accumulation to simulate larger batch sizes.
•	Number of Epochs (3–5): A small number of epochs helps prevent overfitting while ensuring sufficient task adaptation.
•	Learning Rate Scheduler ("cosine"): Smooth decay schedule to fine-tune learning dynamics and avoid abrupt convergence.
•	Gradient Accumulation Steps (4): Allows larger effective batch size without exceeding memory limits.
•	Maximum Sequence Length (512): Balances coverage of multi-step instructions with training efficiency.
These settings are chosen to ensure generalization, training stability, and compatibility with resource-constrained environments.

### Anticipated Challenges and Mitigation Strategies
Several challenges are expected during the fine-tuning process:
1.	Overfitting: Given the domain-specific nature of structured instructions, overfitting is a key concern. This will be mitigated through:
o	Early stopping based on validation loss,
o	Incorporation of dropout layers within adapters,
o	Conservative learning rates,
o	Use of diverse, domain-rich validation sets.
2.	Catastrophic Forgetting: While QLoRA minimizes this risk by freezing base model weights, additional mitigation includes optionally freezing the initial transformer layers to retain foundational language representations.
3.	Computational Constraints: To operate within hardware limitations, 4-bit quantized models will be used alongside QLoRA. Fine-tuning will be conducted on cloud platforms or via Colab Pro+ environments, ensuring access to high-memory GPUs when needed.


# 4.	Evaluation and Benchmarking: 

To evaluate the performance of the fine-tuned LLM in generating structured, task-oriented instructions, both quantitative and qualitative evaluation frameworks will be employed.
### 1. Metrics for Evaluation
Relevant metrics include:
•	BLEU, ROUGE, and METEOR: These measure n-gram and semantic overlap between generated instructions and reference outputs. They are suitable for capturing structural and linguistic fidelity.
•	Exact Match (EM): Percentage of outputs that exactly match gold-standard instruction sets.
•	Step-Level Accuracy: Measures the correctness of individual steps within the multi-step output, especially important in task decomposition scenarios.
These metrics together provide insights into surface-form accuracy, semantic alignment, and logical structure.

### 2. Benchmarking Setup
To assess the effectiveness of fine-tuning, the model will be benchmarked against:
•	The base LLM (e.g., Mistral-7B) without instruction tuning, to measure the gain in task-orientation.
•	Human-generated instructions (gold-standard reference), which serve as an upper-bound for fluency, clarity, and completeness.
•	A held-out validation set containing prompts with diverse intent types and domains, ensuring unbiased evaluation across general use cases.
A/B testing may be conducted between the fine-tuned model and the baseline in a blind evaluation setup.

### 3. Qualitative and Quantitative Assessment
In addition to automated metrics, a human evaluation component will be introduced, focusing on:
•	Fluency: Is the instruction grammatically correct and naturally phrased?
•	Task Relevance: Does the output address the user’s intended task?
•	Instruction Clarity: Are the steps unambiguous and logically ordered?
Meanwhile, quantitative assessment will include:
•	ROUGE-L for structural similarity, especially in capturing longer dependencies.
•	Instruction completeness, measured by the number of coherent, logically sound steps per instruction.
By integrating both human and automated evaluation pipelines, a holistic understanding of the model’s performance and limitations can be achieved.


