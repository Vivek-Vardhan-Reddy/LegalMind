from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class LegalLLM:

    def __init__(self, model_name="google/flan-t5-small"):

        print("Loading LLM model... (first run may download the model)")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        self.device = torch.device("cpu")
        self.model.to(self.device)

        print("LLM model loaded successfully.")


    def generate(self, context, query):

        context = context[:800]

        prompt = f"""
You are a legal contract analysis assistant.

Analyze the contract context and answer the query.

Query:
{query}

Context:
{context}

IMPORTANT:
Return your answer EXACTLY in this format.

Clause Summary: <short summary>

Risk Level: <Low | Medium | High | Critical>

Explanation: <clear explanation>

Do not add any other text.
"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=150,
            num_beams=4,
            early_stopping=True
        )

        result = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return result