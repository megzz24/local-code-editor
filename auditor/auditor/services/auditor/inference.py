import json
from pathlib import Path

from unsloth import FastLanguageModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

BASE_MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"
LORA_ADAPTER_DIR = PROJECT_ROOT / "outputs" / "lora_adapter"
MAX_SEQ_LENGTH = 1024
SYSTEM_PROMPT = "You are a corporate expense auditor."

MAX_NEW_TOKENS = 256


class ExpenseAuditorInference:
    def __init__(self, adapter_path: "str | Path" = LORA_ADAPTER_DIR):
        print(f"Loading base model ({BASE_MODEL_NAME}) + LoRA adapter ({adapter_path})...")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(adapter_path),
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)

        print("Model loaded and ready for inference.")

    def _build_prompt(self, request: dict) -> str:
        user_content = (
            "Audit the following expense claim against company policy. "
            "Return ONLY a JSON object with keys 'verdict', 'policy_section', "
            "and 'reason'.\n\n"
            f"Claim details:\n{json.dumps(request, indent=2)}"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt

    def _extract_json(self, generated_text: str) -> dict:
        try:
            return json.loads(generated_text)
        except json.JSONDecodeError:
            pass

        start = generated_text.find("{")
        end = generated_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = generated_text[start:end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        return {
            "verdict": "Flagged",
            "policy_section": "",
            "reason": (
                "The fine-tuned model did not return valid JSON. "
                f"Raw output: {generated_text}"
            ),
        }

    def audit(self, request: dict) -> dict:
        prompt = self._build_prompt(request)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # deterministic output
            pad_token_id=self.tokenizer.eos_token_id,
        )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return self._extract_json(generated_text.strip())


if __name__ == "__main__":
    example_request = {
        "employee_level": "L3",
        "location": "Mumbai",
        "business_purpose": "Lunch during a work trip",
        "receipt": {
            "merchant_name": "Cafe Mocha",
            "date": "2026-06-10",
            "currency": "INR",
            "total_amount": 450,
            "items": [{"name": "Sandwich Combo", "quantity": 1, "amount": 450}],
        },
    }

    auditor = ExpenseAuditorInference()
    result = auditor.audit(example_request)
    print(json.dumps(result, indent=2))