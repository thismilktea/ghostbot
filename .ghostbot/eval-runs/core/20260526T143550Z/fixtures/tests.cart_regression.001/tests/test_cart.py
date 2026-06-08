from shop.cart import add_to_cart
import pytest


def test_add_to_cart_accepts_positive_quantity():
    cart = {}
    updated = add_to_cart(cart, "apple", 2)
    assert updated["apple"] == 2


def test_add_to_cart_accumulates_quantity():
    cart = {"apple": 1}
    updated = add_to_cart(cart, "apple", 3)
    assert updated["apple"] == 4


def test_add_to_cart_rejects_zero_quantity():
    cart = {"apple": 1}

    with pytest.raises(ValueError):
        add_to_cart(cart, "banana", 0)


def test_add_to_cart_rejects_negative_quantity():
    cart = {"apple": 1}

    with pytest.raises(ValueError):
        add_to_cart(cart, "banana", -1)

