"""Pydantic schemas for receipt parsing, aligned with CORD-v2 dataset annotations."""

from pydantic import BaseModel, field_validator
from typing import Any, Optional, Union


class LineItem(BaseModel):
    """A single menu item from the receipt."""

    nm: Optional[str] = None  # item name/description
    cnt: Optional[str] = None  # quantity (e.g. "1 x")
    unitprice: Optional[str] = None  # unit price
    price: Optional[str] = None  # total price for this item
    sub: Optional[Any] = None  # subcategory (can be str or dict)
    num: Optional[str] = None  # item number
    discountprice: Optional[str] = None  # discount on this item


class SubTotal(BaseModel):
    """Subtotal section of the receipt."""

    subtotal_price: Optional[Any] = None
    tax_price: Optional[Any] = None
    service_price: Optional[Any] = None
    discount_price: Optional[Any] = None
    etc: Optional[Any] = None  # other charges/adjustments


class Total(BaseModel):
    """Total section of the receipt."""

    total_price: Optional[str] = None
    cashprice: Optional[str] = None
    changeprice: Optional[str] = None
    creditcardprice: Optional[str] = None
    menutype_cnt: Optional[str] = None  # number of distinct item types
    menuqty_cnt: Optional[str] = None  # total quantity of items
    total_etc: Optional[str] = None  # other total info


class Receipt(BaseModel):
    """Full structured receipt, matching CORD-v2 gt_parse format."""

    menu: Union[list[LineItem], LineItem] = []
    sub_total: Optional[SubTotal] = None
    total: Total

    @field_validator("menu", mode="before")
    @classmethod
    def ensure_menu_is_list(cls, v: Any) -> list:
        if isinstance(v, dict):
            return [v]
        return v
