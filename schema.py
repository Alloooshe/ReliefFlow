# Arabic column name → internal English field name
# Strips trailing spaces before matching
COLUMN_MAP = {
    "الرقم التسلسلي": "seq_id",

    # Contact name (father or mother)
    "اسم احد افراد العائلة (أب او ام )": "contact_name",
    "اسم احد افراد العائلة(أب او ام )": "contact_name",
    "اسم احد افراد العائلة(أب او ام)": "contact_name",

    # Phone
    "رقم هاتف للتواصل": "phone",

    # Family size
    "عدد افراد العائلة": "family_size",

    # Address
    "العنوان الحالي بالتفصيل": "address",
    "المدينة والعنوان بالتفصيل": "address",

    # Humanitarian situation / displacement reason (same concept, different labels)
    "الحالة الانسانية او سبب الهجرة والمكان ": "humanitarian_situation",
    "الحالة الانسانية او سبب الهجرة والمكان": "humanitarian_situation",
    "سبب الهجرة والمكان ": "humanitarian_situation",
    "سبب الهجرة والمكان": "humanitarian_situation",

    # Need type
    "نوع الحاجة او سبب الحاجة ": "need_type",
    "نوع الحاجة او سبب الحاجة": "need_type",

    # Intermediary
    "الوسيط (الشخص يلي وصلنا للعيلة)": "intermediary",
    "اسم الشخص الوسيط": "intermediary",

    # Sample-2-specific extras
    "اولوية الحاجة من 1 ل 10": "reported_priority",
    "حاجات اضافية": "additional_needs",
    "اسم المتبرع ": "donor",
    "اسم المتبرع": "donor",
    "في حال كان هناك لحاجة دعم مالي مع السلة الغذائية ولماذا؟": "financial_support_note",
}

# Columns that must have a value for a row to be considered real data
IDENTITY_COLS = ["contact_name", "family_size", "need_type", "humanitarian_situation"]

# Known city names used for extraction from free-text addresses
KNOWN_CITIES = [
    "حمص", "مصياف", "صافيتا", "طرطوس", "دمشق", "جبلة", "حلب",
    "اللاذقية", "تدمر", "القصير", "بانياس", "السويداء", "درعا",
]
