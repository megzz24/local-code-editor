from paddleocr import PaddleOCR


class PaddleOCRService:
    def __init__(self):
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def extract_text(self, image_path: str) -> str:
        result = self.ocr.predict(image_path)
        if not result:
            return ""

        page = result[0]
        texts = page.get("rec_texts", [])
        scores = page.get("rec_scores", [])
        polys = page.get("rec_polys", [])

        boxes_data = []
        for text, score, poly in zip(texts, scores, polys):
            avg_y = sum(point[1] for point in poly) / len(poly)
            min_x = min(point[0] for point in poly)
            boxes_data.append(
                {
                    "text": text,
                    "score": score,
                    "avg_y": avg_y,
                    "min_x": min_x,
                }
            )

        ROW_THRESHOLD = 20  
        boxes_data.sort(key=lambda x: x["avg_y"])

        rows = []
        for item in boxes_data:
            placed = False
            for row in rows:
                if abs(item["avg_y"] - row["avg_y"]) <= ROW_THRESHOLD:
                    row["items"].append(item)
                    row["avg_y"] = (
                        row["avg_y"] * (len(row["items"]) - 1)
                        + item["avg_y"]
                    ) / len(row["items"])
                    placed = True
                    break
            if not placed:
                rows.append(
                    {
                        "avg_y": item["avg_y"],
                        "items": [item],
                    }
                )

        receipt_lines = []
        for row in rows:
            row["items"].sort(key=lambda x: x["min_x"])
            line_text = " ".join(item["text"] for item in row["items"])
            receipt_lines.append(line_text)

        extracted_text = "\n".join(receipt_lines)

        print("\n===== OCR OUTPUT =====")
        print(extracted_text)
        print("======================\n")

        return extracted_text


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m services.paddle_service <path_to_receipt_image>")
    else:
        service = PaddleOCRService()
        text = service.extract_text(sys.argv[1])
        print(text)