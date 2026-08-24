from models import User, Product
from cart import Cart

class Order:
    def __init__(self, order_id: int, user: User, cart: Cart):
        self.order_id = order_id
        self.user = user
        self.cart = cart
        self.status = "pending"

    def place(self, products: dict[int, Product]) -> dict:
        if not self.cart.items:
            raise ValueError("Cart is empty")
        total = sum(
            products[pid].price * qty
            for pid, qty in self.cart.items.items()
            if pid in products
        )
        self.status = "placed"
        return {"order_id": self.order_id, "user": self.user.full_name(), "total": total, "status": self.status}

    def cancel(self):
        if self.status == "placed":
            self.status = "cancelled"
        else:
            raise ValueError(f"Cannot cancel order with status: {self.status}")


def get_order_summary(order: Order, products: dict[int, Product]) -> str:
    lines = [f"Order #{order.order_id} for {order.user.full_name()}"]
    for pid, qty in order.cart.items.items():
        if pid in products:
            p = products[pid]
            lines.append(f"  - {p.name} x{qty} @ ${p.price:.2f}")
    lines.append(f"Status: {order.status}")
    return "\n".join(lines)