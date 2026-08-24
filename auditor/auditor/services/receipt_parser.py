import json
from pathlib import Path
from langchain_ollama import ChatOllama


class ReceiptParser:
    def __init__(self):
        self.llm = ChatOllama(
            model="qwen3:8b",
            temperature=0,
            format="json",
        )

        prompt_path = (
            Path(__file__).parent.parent
            / "prompts"
            / "receipt_prompt.txt"
        )
        self.prompt_template = prompt_path.read_text()

    def parse(self, receipt_text: str) -> dict:
        prompt = self.prompt_template.format(receipt_text=receipt_text)

        response = self.llm.invoke(prompt)
        content = response.content

        if not isinstance(content, str):
            raise ValueError(
                f"Expected string response but got {type(content)}"
            )

        try:
            receipt = json.loads(content)
        except json.JSONDecodeError:
            print(content)
            raise

        for item in receipt.get("items", []):
            try:
                item["quantity"] = int(item.get("quantity", 1))
            except Exception:
                item["quantity"] = 1
            try:
                item["amount"] = float(item.get("amount", 0))
            except Exception:
                item["amount"] = 0.0

        try:
            receipt["total_amount"] = float(receipt.get("total_amount", 0))
        except Exception:
            receipt["total_amount"] = 0.0

        return receipt


if __name__ == "__main__":
    sample_text = (
        "CAFE MOCHA\n"
        "Date: 01/06/2026\n"
        "Sandwich 1 250\n"
        "Coffee 2 300\n"
        "TOTAL INR 550\n"
    )
    parser = ReceiptParser()
    result = parser.parse(sample_text)
    print(json.dumps(result, indent=2))