class User:
    def __init__(self, user_id: int, name: str, email: str):
        self.user_id = user_id
        self.name = name
        self.email = email

    def full_name(self) -> str:
        return self.name.strip().title()

    def is_valid_email(self) -> bool:
        return "@" in self.email and "." in self.email.split("@")[-1]


class Product:
    def __init__(self, id: int, name: str, price: float, stock: int):
        self.id = id
        self.name = name
        self.price = price
        self.stock = stock

    def is_in_stock(self) -> bool:
        return self.stock > 0

    def apply_discount(self, percent: float) -> float:
        return self.price * (1 - percent / 100)