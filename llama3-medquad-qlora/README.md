---
library_name: transformers
tags:
- medical
- qlora
- llama-3
- finetuned
- question-answering
---

# LLaMA-3 8B Instruct - MedQuad Medical QnA (QLoRA)

This model is a fine-tuned version of **LLaMA-3 8B Instruct** using **QLoRA (4-bit quantization + LoRA adapters)** on the **MedQuad Medical QnA Dataset**.  
It is designed to answer **medical domain questions** across various categories like treatment, symptoms, causes, prevention, inheritance, etc.

---

## Model Details

### Model Description
- **Base model:** [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)  
- **Fine-tuning method:** QLoRA (4-bit quantization with LoRA adapters)  
- **Task:** Medical Question Answering (Instruction-tuned style)  
- **Languages:** English  
- **Framework:** 🤗 Transformers, PEFT, TRL  
- **Quantization:** 4-bit (nf4, bfloat16 compute)  
- **License:** [Llama 3 license](https://ai.meta.com/llama/license/)  

### Developers
- **Developed by:** Arush Pettem  
- **Dataset:** [keivalya/MedQuad-MedicalQnADataset](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset)  

---

## Model Sources
- **Repository:** [Your Hugging Face repo link]  
- **Paper:** ["MedQuAD: Medical Question Answering Dataset"](https://academic.oup.com/database/article/doi/10.1093/database/bay068/5058107)  
- **Demo:** (Optional if you make a Gradio Space)  

---

## Uses

### Direct Use
- Answering medical questions in categories such as treatment, symptoms, causes, prevention, outlook, etc.  
- Educational and research purposes in healthcare QA systems.  

### Downstream Use
- Integration into healthcare chatbots.  
- Fine-tuning on domain-specific sub-corpora (e.g., cardiology QnA).  
- Evaluation for explainable AI in medical NLP.  

### Out-of-Scope Use
⚠️ This model is **not a substitute for professional medical advice**. It should **not be used for clinical decision-making or diagnosis**.  

---

## Bias, Risks, and Limitations
- **Bias:** Model inherits potential biases from MedQuad and the LLaMA base model.  
- **Risks:** Incorrect or incomplete medical answers may mislead users if used in real-world clinical contexts.  
- **Limitations:** Trained on static QA pairs, so may not generalize to open-ended patient conversations.  

### Recommendations
- Use in **controlled, educational, or research settings** only.  
- Always validate outputs with trusted medical sources.  

---

## How to Get Started with the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model = AutoModelForCausalLM.from_pretrained("your-username/llama3-medquad-qlora")
tokenizer = AutoTokenizer.from_pretrained("your-username/llama3-medquad-qlora")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

query = "What are the symptoms of asthma?"
print(pipe(query, max_new_tokens=100))
