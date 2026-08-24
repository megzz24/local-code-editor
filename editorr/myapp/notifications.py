from models import User

def send_email(user: User, subject: str, body: str) -> bool:
    if not user.is_valid_email():
        return False
    print(f"Sending to {user.email}: [{subject}] {body}")
    return True

def notify_order_placed(user: User, order_id: int) -> bool:
    return send_email(
        user,
        subject=f"Order confirmed for {user.full_name()}",
        body=f"Your order #{order_id} has been placed successfully.",
    )

def notify_order_cancelled(user: User, order_id: int) -> bool:
    return send_email(
        user,
        subject="Order cancelled",
        body=f"Your order #{order_id} has been cancelled.",
    )