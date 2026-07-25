"""Category/subcategory/brand taxonomy -- pure data, no sampling logic.

Shared between data_gen/ (uses the full structure, including
subcategories/price_range/tags, to generate synthetic items) and
pipelines/ (uses only CATEGORIES/ALL_BRANDS, to validate that the daily
feature aggregation never emits an unknown category or brand). The
taxonomy is a fixed, hand-authored structure so categories, subcategories,
and brands form a believable retail hierarchy rather than arbitrary
combinations.
"""

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
