from models import User, Product

_users: dict[int, User] = {}
_products: dict[int, Product] = {}

def save_user(user: User):
    _users[user.user_id] = user

def get_user(user_id: int) -> User | None:
    return _users.get(user_id)

def save_product(product: Product):
    _products[product.id] = product

def get_product(id: int) -> Product | None:
    return _products.get(id)

def get_all_products() -> list[Product]:
    return list(_products.values())

def decrement_stock(id: int, quantity: int):
    product = _products.get(id)
    if product is None:
        raise KeyError(f"Product {id} not found")
    if product.stock < quantity:
        raise ValueError(f"Insufficient stock for product {id}")
    product.stock -= quantity