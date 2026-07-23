"""Category/subcategory/brand taxonomy and pure sampling functions.

The taxonomy is a fixed, hand-authored structure (not randomly generated)
so that categories, subcategories, and brands form a believable retail
hierarchy rather than arbitrary combinations.
"""

import numpy as np

TAXONOMY = {
    "Footwear": {
        "subcategories": ["Sneakers", "Boots", "Sandals"],
        "brands": ["Nike", "Adidas", "New Balance", "Vans"],
        "price_range": (30.0, 220.0),
        "tags": ["lightweight", "waterproof", "casual", "athletic", "leather"],
    },
    "Electronics": {
        "subcategories": ["Headphones", "Smartwatches", "Chargers"],
        "brands": ["Sony", "Anker", "JBL", "Samsung"],
        "price_range": (15.0, 400.0),
        "tags": ["wireless", "noise-cancelling", "fast-charging", "compact"],
    },
    "Apparel": {
        "subcategories": ["T-Shirts", "Jackets", "Jeans"],
        "brands": ["Levi's", "Uniqlo", "Patagonia", "Champion"],
        "price_range": (12.0, 180.0),
        "tags": ["cotton", "slim-fit", "vintage", "outdoor", "breathable"],
    },
    "Home": {
        "subcategories": ["Cookware", "Bedding", "Lighting"],
        "brands": ["IKEA", "Lodge", "Philips", "Brooklinen"],
        "price_range": (10.0, 250.0),
        "tags": ["minimalist", "handmade", "durable", "eco-friendly"],
    },
    "Beauty": {
        "subcategories": ["Skincare", "Haircare", "Fragrance"],
        "brands": ["CeraVe", "Olaplex", "The Ordinary", "Dove"],
        "price_range": (8.0, 120.0),
        "tags": ["fragrance-free", "hydrating", "natural", "cruelty-free"],
    },
    "Sports": {
        "subcategories": ["Yoga Gear", "Cycling", "Camping"],
        "brands": ["Coleman", "Trek", "Lululemon", "REI"],
        "price_range": (10.0, 500.0),
        "tags": ["outdoor", "lightweight", "durable", "packable"],
    },
    "Toys": {
        "subcategories": ["Building Sets", "Board Games", "Puzzles"],
        "brands": ["LEGO", "Hasbro", "Mattel", "Ravensburger"],
        "price_range": (5.0, 150.0),
        "tags": ["educational", "family", "collectible"],
    },
    "Books": {
        "subcategories": ["Fiction", "Nonfiction", "Children's"],
        "brands": ["Penguin", "Scholastic", "HarperCollins", "Vintage"],
        "price_range": (6.0, 45.0),
        "tags": ["bestseller", "paperback", "illustrated"],
    },
}

CATEGORIES = list(TAXONOMY.keys())
ALL_BRANDS = sorted({brand for spec in TAXONOMY.values() for brand in spec["brands"]})


class UnknownCategoryError(Exception):
    """Raised when a category is not present in the taxonomy."""


def sample_category(rng: np.random.Generator) -> str:
    """Samples a category uniformly from the taxonomy.

    Args:
        rng: A numpy random Generator.

    Returns:
        str: A category name.
    """
    return rng.choice(CATEGORIES)


def sample_subcategory(rng: np.random.Generator, category: str) -> str:
    """Samples a subcategory belonging to the given category.

    Args:
        rng: A numpy random Generator.
        category: A category name present in TAXONOMY.

    Returns:
        str: A subcategory name.

    Raises:
        UnknownCategoryError: If category is not in the taxonomy.
    """
    if category not in TAXONOMY:
        raise UnknownCategoryError(f"Unknown category: {category!r}")
    return rng.choice(TAXONOMY[category]["subcategories"])


def sample_brand(rng: np.random.Generator, category: str) -> str:
    """Samples a brand belonging to the given category.

    Args:
        rng: A numpy random Generator.
        category: A category name present in TAXONOMY.

    Returns:
        str: A brand name.

    Raises:
        UnknownCategoryError: If category is not in the taxonomy.
    """
    if category not in TAXONOMY:
        raise UnknownCategoryError(f"Unknown category: {category!r}")
    return rng.choice(TAXONOMY[category]["brands"])


def sample_price(rng: np.random.Generator, category: str) -> float:
    """Samples a price within the category's price range.

    Uses a lognormal-shaped draw (clipped to the category's range) so
    prices cluster toward the low end with a long tail of premium
    items, mimicking real retail pricing better than a uniform draw.

    Args:
        rng: A numpy random Generator.
        category: A category name present in TAXONOMY.

    Returns:
        float: A price rounded to 2 decimal places.

    Raises:
        UnknownCategoryError: If category is not in the taxonomy.
    """
    if category not in TAXONOMY:
        raise UnknownCategoryError(f"Unknown category: {category!r}")
    low, high = TAXONOMY[category]["price_range"]
    raw = rng.lognormal(mean=0.0, sigma=0.5)
    normalized = min(raw / 4.0, 1.0)
    price = low + normalized * (high - low)
    return round(price, 2)


def sample_tags(rng: np.random.Generator, category: str) -> list[str]:
    """Samples 2-4 tags without replacement from the category's tag pool.

    Args:
        rng: A numpy random Generator.
        category: A category name present in TAXONOMY.

    Returns:
        list[str]: Between 2 and 4 tag strings.

    Raises:
        UnknownCategoryError: If category is not in the taxonomy.
    """
    if category not in TAXONOMY:
        raise UnknownCategoryError(f"Unknown category: {category!r}")
    pool = TAXONOMY[category]["tags"]
    n = int(rng.integers(2, min(4, len(pool)) + 1))
    return list(rng.choice(pool, size=n, replace=False))
