# Single merged anonymized file (relative to app working directory)
DATA_FILE = "data_anonymized/relief_data.xlsx"

# Sheet names inside DATA_FILE
SHEET_MAIN    = "main"
SHEET_MEMBERS = "members"
SHEET_DAMAGE  = "damage"
SHEET_NEEDS   = "needs"


# Arabic column names for main CSV
class MainCols:
    ID = "_id"
    GOVERNORATE = "المحافظة"
    DISTRICT = "المنطقة الإدارية"
    SUB_DISTRICT = "المنطقة الفرعية"
    VILLAGE = "اسم القرية"
    DISPLACEMENT_TYPE = "طبيعة العائلة"
    HOUSING_TYPE = "طبيعة السكن"
    FAMILY_SIZE = "عدد أفراد العائلة الفعلي"
    DEPENDENTS = "عدد الأفراد المعالين"
    BREADWINNERS = "عدد الأفراد المعيلين"
    ACCESS_TYPE = "طبيعة الوصول"
    SURVEY_DATE = "تاريخ النشاط"


# Arabic column names for members CSV
class MemberCols:
    FAMILY_ID = "_submission__id"
    RELATION = "علاقته بالعائلة"
    IS_HOH = "هل هذا الشخص هو رب الأسرة"
    MARITAL_STATUS = "الحالة الاجتماعية"
    GENDER = "الجنس"
    INDIVIDUAL_STATUS = "حالة الفرد"
    OCCUPATION = "occupation_bucket"
    BIRTH_YEAR = "birth_year"
    ASSAULT = "هل تعرضت لأي إعتداء سواء كان لفظي أو لفظي"
    V_IMMEDIATE_HEALTH = "نقاط الضعف/حالة صحية تطلب تدخل فوري"
    V_CHRONIC_DISEASE = "نقاط الضعف/مرض مزمن"
    V_MENTAL_DISABILITY = "نقاط الضعف/إعاقة عقلية"
    V_PHYSICAL_DISABILITY = "نقاط الضعف/إعاقة جسدية"
    V_SPEECH_IMPAIRMENT = "نقاط الضعف/خلل في النطق"
    V_HEARING_IMPAIRMENT = "نقاط الضعف/خلل في السمع"
    V_VISION_IMPAIRMENT = "نقاط الضعف/خلل في الرؤية"
    V_ELDERLY_WITH_CHILDREN = "نقاط الضعف/مسن مع أطفال"
    V_ELDERLY_UNABLE = "نقاط الضعف/مسن غير قادر"
    V_CHILD_PARENT = "نقاط الضعف/طفل والد"
    V_MARRIED_CHILD = "نقاط الضعف/طفل متزوج"
    V_CHILD_CAREGIVER = "نقاط الضعف/طفل مقدم رعاية"
    V_CHILD_LABORER = "نقاط الضعف/طفل عامل"
    V_CHILD_DROPOUT = "نقاط الضعف/طفل متسرب من المدرسة"
    V_CHILD_SPECIAL_ED = "نقاط الضعف/طفل لديه احتياجات تعليمية خاصة"
    V_SEPARATED_CHILD = "نقاط الضعف/طفل منفصل"
    V_UNACCOMPANIED_CHILD = "نقاط الضعف/طفل غير مصحوب"
    V_FEMALE_HOH = "نقاط الضعف/امرأة ربة اسرة"
    V_SINGLE_FATHER = "نقاط الضعف/اب مقدم رعاية وحيد"
    V_UNDOCUMENTED = "نقاط الضعف/شخص لا يمتلك وثائق"


# Arabic column names for damage CSV
class DamageCols:
    FAMILY_ID = "_submission__id"
    CATEGORY = "فئة الضرر"
    TYPE = "نوع الضرر"
    CLASSIFICATION = "تصنيف الضرر"
    QUANTITY = "الكمية المرتبطة بالضرر"


# Arabic column names for needs CSV
class NeedsCols:
    FAMILY_ID = "_submission__id"
    PROGRAM = "البرنامج (التصنيف الأول)"
    CLASSIFICATION = "تصنيف الاحتياج (التصنيف الثاني)"
    QUANTITY = "الكمية"
    SERVICE_RECEIVED = "هل حصلت العائلة (أو الفرد) على الخدمة"


# Displacement type values
DISPLACED_TYPES = frozenset({
    "نازح داخلي",
    "نازح عائد (كان في منطقة أخرى داخل سوريا)",
    "لاجئ عائد (من خارج القطر)",
})

# Housing type values
HOUSING_RENTING = "أجار"
HOUSING_HOMELESS = "لا يوجد مسكن"

# Needs classification to need_type string
BABY_NEEDS = frozenset({
    "حليب أطفال",
    "حفاضات أطفال",
    "مكملات غذائية للرضع",
})

# Needs program mapping to English categories
PROGRAM_TO_CATEGORY = {
    "احتياجات غذائية": "food",
    "احتياجات غير غذائية أو نظافة": "hygiene",
    "احتياجات طبية أو صحية": "medical",
    "احتياجات مالية أو سبل العيش": "financial",
    "الحماية (CP,GBV,PSS)": "psychosocial",
    "التعليم": "education",
    "قانوني": "legal",
    "احتياجات مأوى أو صرف صحي": "shelter",
}
