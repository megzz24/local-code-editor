from models import Product

class Cart:
    def __init__(self):
        self.items: dict[int, int] = {}

    def add_item(self, product: Product, quantity: int = 1):
        if not product.is_in_stock():
            raise ValueError(f"{product.name} is out of stock")
        self.items[product.id] = self.items.get(product.id, 0) + quantity

    def remove_item(self, id: int, quantity: int = 1):
        if id in self.items:
            if self.items[id] <= quantity:
                del self.items[id]
            else:
                self.items[id] -= quantity

    def total_items(self) -> int:
        return sum(self.items.values())

    def clear(self):
        self.items = {}