# Expected Data Format

Place **4 CSV files** exported from KoBoToolbox in this folder.
The app reads them via `anonymize.py` → `ingest.py`.

## File Naming

The app auto-detects files by name pattern:

| File | Detected by | Rows (example) |
|------|-------------|----------------|
| `*الجمعية*.csv` or any non-matching name | Main registration sheet | ~7,200 |
| `*FamilyMembers*.csv` or `*member*` | Family members | ~19,200 |
| `*Damag*.csv` | Damage records | ~7,600 |
| `*Need*.csv` | Needs records | ~17,200 |

## Step 1 — Run Anonymizer

Before loading the app, strip PII from raw CSVs:

```bash
python anonymize.py
# Writes 4 clean CSVs to data_anonymized/
```

---

## Table Schemas

### 1. Main Registration Sheet (one row per family)

Join key: `_id` (parent)

| Column (Arabic) | Description | Type |
|---|---|---|
| `تاريخ النشاط` | Survey / activity date | date |
| `المحافظة` | Governorate | text (طرطوس / اللاذقية) |
| `طبيعة الوصول` | Access type | text |
| `المنطقة الإدارية` | Administrative district | text |
| `المنطقة الفرعية` | Sub-district | text |
| `اسم القرية` | Village name | text |
| `طبيعة العائلة` | Displacement type | categorical¹ |
| `طبيعة السكن` | Housing type | categorical² |
| `عدد أفراد العائلة الفعلي` | Actual family size | integer |
| `عدد الأفراد المعالين` | Number of dependents | integer |
| `عدد الأفراد المعيلين` | Number of breadwinners | integer |
| `_id` | KoBoToolbox submission ID | integer (join key) |
| `_submission_time` | Submission timestamp | datetime |

¹ Displacement type values: `نازح داخلي` · `نازح عائد (كان في منطقة أخرى داخل سوريا)` · `لاجئ عائد (من خارج القطر)` · `مجتمع مضيف`  
² Housing type values: `ملك` · `أجار` · `أستضافة` · `لا يوجد مسكن` · `أخرى`

**Columns removed by anonymizer (PII):**
`رقم هاتف للعائلة` · `مدخل البيانات` · `رقم هاتف مدخل البيانات` · `العنوان التفصيلي` · `رقم دفتر العائلة` · `ملاحظات عن العائلة` · `المركز المجتمعي` · `عنوان العائلة الأصلي` · `_uuid` · `_submitted_by`

---

### 2. FamilyMembers.csv (one row per individual)

Join key: `_submission__id` → `_id` in main

| Column (Arabic) | Description | Type |
|---|---|---|
| `علاقته بالعائلة` | Relation to family head | categorical³ |
| `هل هذا الشخص هو رب الأسرة` | Is head of household? | نعم / لا |
| `الحالة الاجتماعية` | Marital status | categorical⁴ |
| `الجنس` | Gender | ذكر / أنثى |
| `حالة الفرد` | Individual status | categorical⁵ |
| `نقاط الضعف/...` | Vulnerability flags (21 columns) | 0.0 / 1.0 |
| `هل تعرضت لأي إعتداء` | Experienced assault (yes/no) | text |
| `occupation_bucket` | Bucketed occupation (added by anonymizer) | categorical⁶ |
| `birth_year` | Birth year (derived from DOB by anonymizer) | integer |
| `_submission__id` | Parent family ID | integer (join key) |

³ Relation values: `الزوج` · `الزوجة` · `الابن` · `الابنة` · `الجد` · `الجدة` · `الحفيد` · `الحفيدة` · `غير قرابة`  
⁴ Marital status: `أعزب/عزباء` · `متزوج/ة` · `مطلق/ة` · `منفصل/ة`  
⁵ Individual status: `موجود` · `مفقود` · `متوفي` · `مصاب` · `اخرى`  
⁶ Occupation buckets: `no_work` · `homemaker` · `student` · `agriculture` · `employed` · `self_employed` · `military` · `retired` · `child` · `other` · `unknown`

**Columns removed by anonymizer (PII):**
`الاسم الاول` · `الكنية` · `اسم الاب` · `اسم الام` · `الرقم الوطني` · `رقم هاتف الفرد` · `تفاصيل الاعتداء` · `تاريخ الميلاد` (→ birth_year kept) · `العمل السابق` · `_parent_table_name`

---

### 3. Damag_Group.csv (one row per damage record)

Join key: `_submission__id` → `_id` in main

| Column (Arabic) | Description | Values |
|---|---|---|
| `فئة الضرر` | Damage category | أضرار في الممتلكات · أضرار في الثروة الزراعية · أضرار في الثروة الحيوانية |
| `نوع الضرر` | Damage type | حرق أو تدمير · سرقات · استيلاء · محاصيل · أشجار مثمرة · ماشية · تجهيزات |
| `تصنيف الضرر` | Damage classification | بيت · سيارة · ذهب · أموال شخصية · أجهزة الكترونية · أغنام · أبقار · ... |
| `الكمية المرتبطة بالضرر` | Quantity | numeric |
| `_submission__id` | Parent family ID | integer (join key) |

**Columns removed by anonymizer:**
`معلومات تفصيلية عن الضرر` (free-text) · `_parent_table_name`

---

### 4. Needsgroup.csv (one row per need record)

Join key: `_submission__id` → `_id` in main

| Column (Arabic) | Description | Values |
|---|---|---|
| `البرنامج (التصنيف الأول)` | Need program (level 1) | احتياجات غذائية · احتياجات طبية · احتياجات مأوى · احتياجات مالية · الحماية · التعليم · قانوني · احتياجات غير غذائية |
| `تصنيف الاحتياج (التصنيف الثاني)` | Need classification (level 2) | بطانيات · فرشات · البسة · تأهيل منزل · طعام · احتياج دواء · دعم مالي · جلسات دعم نفسي · ... |
| `الكمية` | Quantity | numeric |
| `هل حصلت العائلة (أو الفرد) على الخدمة` | Service received? | نعم / لا |
| `_submission__id` | Parent family ID | integer (join key) |

**Columns removed by anonymizer:**
`تفاصيل الاحتياج` (free-text) · `تحديد الاحتياج` (free-text) · `_parent_table_name`

---

## Relational Structure

```
main (_id)
 ├── FamilyMembers (_submission__id → _id)
 ├── Damag_Group   (_submission__id → _id)
 └── Needsgroup    (_submission__id → _id)
```

`ingest.py` joins all 4 tables and aggregates to one flat row per family.
