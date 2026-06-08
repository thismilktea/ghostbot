def add_to_cart(cart: dict[str, int], item: str, quantity: int) -> dict[str, int]:
    updated = dict(cart)
    updated[item] = updated.get(item, 0) + quantity
    return updated
