"""
payslip_analyzer.py - Malaysian Payslip Analyzer
OCR:         NVIDIA NIM (llama-4-maverick)
Explanation: OpenRouter (nemotron-3-super-120b)
Advice:      OpenRouter (gpt-oss-120b) — different rate limit bucket
"""

import sys, os, json, base64, re, requests, cv2, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NIM_API_KEY        = os.environ.get("NIM_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
NIM_BASE_URL        = "https://integrate.api.nvidia.com/v1/chat/completions"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

OCR_MODEL         = "meta/llama-4-maverick-17b-128e-instruct"
EXPLANATION_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"   # for payslip explanation
ADVICE_MODEL      = "openai/gpt-oss-120b:free" # for financial advice


def strip_thinking(text: str) -> str:
    """Remove reasoning traces from model output. Keep only the final answer."""
    import re as _re
    text = _re.sub(r'<think>.*?</think>', '', text, flags=_re.DOTALL)
 
    reasoning_patterns = [
        r"^We need to.*$",
        r"^Let'?s (compute|calculate|check|craft|count|think|see|recall|verify|answer|produce|aim|avoid|make sure).*$",
        r"^Check for.*$",
        r"^Compute.*$",
        r"^Solve for.*$",
        r"^Approx.*$",
        r"^Actually.*check.*$",
        r"^Must avoid.*$",
        r"^So we (need|just|can|have|must).*$",
        r"^Thus.*$",
        r"^Then.*payment.*$",
        r"^Use.*formula.*$",
        r"^Make sure.*$",
        r"^Total words.*$",
        r"^r =.*$",
        r"^n =.*$",
        r"^P =.*$",
        r"^Monthly.*instalment.*=.*$",
        r".*not present.*$",
        r".*inadvertently.*$",
        r".*substring.*$",
        r".*Malay word.*$",
        r".*presumably.*$",
    ]
 
    clean_lines = []
    for line in text.splitlines():
        skip = any(_re.match(pat, line.strip(), _re.IGNORECASE) for pat in reasoning_patterns)
        if not skip:
            clean_lines.append(line)
 
    result = _re.sub(r'\n{3,}', '\n\n', '\n'.join(clean_lines))
    return result.strip()
 
 
 
# ─────────────────────────────────────────
# STEP 1 — IMAGE PREPROCESSING
# ─────────────────────────────────────────
def preprocess_image(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_warning = cv2.Laplacian(gray, cv2.CV_64F).var() < 100
 
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
 
    edges = cv2.Canny(enhanced, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    angle = 0.0
    if lines is not None:
        angles = [np.degrees(t) - 90 for r, t in lines[:, 0]
                  if abs(np.degrees(t) - 90) < 10]
        if angles:
            angle = float(np.median(angles))
 
    if abs(angle) > 0.5:
        h, w = enhanced.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        enhanced = cv2.warpAffine(enhanced, M, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
 
    preprocessed_path = image_path.replace(".", "_preprocessed.")
    cv2.imwrite(preprocessed_path, enhanced)
    return preprocessed_path, blur_warning
 
 
# ─────────────────────────────────────────
# STEP 2 — SMART OCR + EXTRACTION via NVIDIA NIM
# ─────────────────────────────────────────
def extract_fields_with_ai(image_path: str) -> dict:
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
 
    ext = image_path.split(".")[-1].lower()
    mime = "image/jpeg" if ext in ["jpg", "jpeg"] else "image/png"
 
    prompt = """You are an expert at reading Malaysian payslips (slip gaji).
This payslip may be in English, Bahasa Malaysia, or a mix of both.
Different companies use different field names and layouts.
 
Your job is to identify and extract the correct values regardless of how they are labelled.
 
Common field name variations in Malaysian payslips:
- Employee name: NAME, NAMA, NAMA PEKERJA, EMPLOYEE, PEKERJA, NAMA PENUH
- IC number: IC, NO. IC, NRIC, NO. KAD PENGENALAN, MYKAD, I/C NO
- Company: COMPANY, SYARIKAT, MAJIKAN, EMPLOYER, SDN BHD, BERHAD
- Period: MONTH, BULAN, PERIOD, TEMPOH, PAY DATE, TARIKH, END PAYMENT, UNTUK BULAN, SALARY STATEMENT FOR THE MONTH OF
- Basic pay: BASIC, GAJI ASAS, BASIC PAY, BASIC SALARY, GAJI POKOK, BASE RATE
- Gross pay: GROSS, JUMLAH PENDAPATAN, GROSS PAY, GROSS SALARY, PENDAPATAN KASAR, TOTAL EARNINGS, TOTAL INCOME
- Net pay: NET PAY, GAJI BERSIH, TAKE HOME, BERSIH, NET SALARY, JUMLAH BERSIH
- EPF/KWSP: EPF, KWSP, KUMPULAN WANG SIMPANAN PEKERJA, EMPLOYEE EPF, CARUMAN KWSP
- SOCSO/PERKESO: SOCSO, PERKESO
- EIS/SIP: EIS, SIP, INSURANS PEKERJAAN
- PCB/Tax: PCB, POTONGAN CUKAI BERJADUAL, INCOME TAX, CUKAI PENDAPATAN, LHDN, TAX
- Total deduction: TOTAL DEDUCTION, JUMLAH POTONGAN, TOTAL POTONGAN, TOTAL DEDUCTIONS
 
Extract ALL values and return ONLY a valid JSON object:
{
  "employer_name": "company name or null",
  "employee_name": "full name of person (NOT employee ID/number) or null",
  "employee_id": "employee ID/number or null",
  "employee_ic": "Malaysian IC number e.g. 901234-56-7890 or null",
  "period": "pay period month and year e.g. November 2021 or null",
  "position": "job title or null",
  "basic_salary": number or null,
  "gross_salary": number or null,
  "net_pay": number or null,
  "epf_employee": number or null,
  "epf_employer": number or null,
  "socso_employee": number or null,
  "eis_employee": number or null,
  "pcb": number or null,
  "total_deduction": number or null,
  "other_allowances": {"allowance_name": amount},
  "other_deductions": {"deduction_name": amount}
}
 
Rules:
- employee_name must be the PERSON'S FULL NAME, never an ID
- employee_ic must be the IC number only, format XXXXXX-XX-XXXX or null
- All amounts must be plain numbers, no RM or commas
- If not found, use null
- Return ONLY the JSON, nothing else"""
 
    payload = {
        "model": OCR_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mime};base64,{image_data}"}},
                    {"type": "text", "text": prompt}
                ]
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.1
    }
 
    headers = {
        "Authorization": f"Bearer {NIM_API_KEY}",
        "Content-Type": "application/json"
    }
 
    response = requests.post(NIM_BASE_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    raw = response.json()["choices"][0]["message"]["content"]
    print(f"   Raw AI response:\n{raw}\n")
 
    try:
        clean = re.sub(r'```(?:json)?', '', raw).strip()
        fields = json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            fields = json.loads(match.group())
        else:
            raise ValueError("Could not parse AI response as JSON")
 
    normalized = {
        "employer_name":   fields.get("employer_name") or "Not detected",
        "employee_name":   fields.get("employee_name") or "Not detected",
        "employee_id":     fields.get("employee_id"),
        "employee_ic":     fields.get("employee_ic"),
        "period":          fields.get("period") or "Not detected",
        "position":        fields.get("position"),
        "basic_salary":    _to_float(fields.get("basic_salary")),
        "gross_salary":    _to_float(fields.get("gross_salary")),
        "net_pay":         _to_float(fields.get("net_pay")),
        "epf_employee":    _to_float(fields.get("epf_employee")),
        "epf_employer":    _to_float(fields.get("epf_employer")),
        "socso":           _to_float(fields.get("socso_employee") or fields.get("socso")),
        "eis":             _to_float(fields.get("eis_employee") or fields.get("eis")),
        "pcb":             _to_float(fields.get("pcb")),
        "total_deduction": _to_float(fields.get("total_deduction")),
        "other_allowances": fields.get("other_allowances", {}),
        "other_deductions": fields.get("other_deductions", {}),
        "meal_allow":      None,
        "travel_allow":    None,
    }
 
    for k, v in (fields.get("other_allowances") or {}).items():
        if "meal" in k.lower() or "makan" in k.lower():
            normalized["meal_allow"] = _to_float(v)
        if "travel" in k.lower() or "transport" in k.lower() or "pengangkutan" in k.lower():
            normalized["travel_allow"] = _to_float(v)
 
    return normalized
 
 
def _to_float(val):
    if val is None:
        return None
    try:
        if isinstance(val, str):
            val = val.replace(",", "").replace("RM", "").strip()
        return float(val)
    except (ValueError, TypeError):
        return None

def _get_age_from_ic(ic: str) -> int | None:
    """Extract age from Malaysian IC (YYMMDD-SS-SSSS)"""
    try:
        ic_clean = ic.replace("-", "").strip()
        yy = int(ic_clean[:2])
        mm = int(ic_clean[2:4])
        dd = int(ic_clean[4:6])
        year = (2000 + yy) if yy <= 25 else (1900 + yy)
        from datetime import date
        dob = date(year, mm, dd)
        today = date.today()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    except:
        return None
 
 
# ─────────────────────────────────────────
# STEP 3 — VALIDATE DEDUCTIONS
# ─────────────────────────────────────────
def validate_deductions(fields: dict) -> dict:
    gross = fields.get("gross_salary") or 0
    basic = fields.get("basic_salary") or gross
    validations = {}
 
    # EPF: 11% of basic salary
    if fields.get("epf_employee") and basic > 0:
        actual = fields["epf_employee"]
        ic = fields.get("employee_ic")
        age = _get_age_from_ic(ic) if ic else None
        print(f"DEBUG IC: {ic}, Age: {age}")
        above_60 = age >= 60 if age is not None else False

        if above_60:
            expected = round(basic * 0.055, 2)
            rate_label = "5.5% (age 60+)"
        else:
            expected = round(basic * 0.11, 2)
            rate_label = "11%"

        tolerance = max(basic * 0.02, 5)
        validations["epf"] = {
            "actual": round(actual, 2),
            "expected_on_basic": expected,
            "status": "✅ Correct" if abs(actual - expected) <= tolerance else "⚠️ Discrepancy",
            "note": f"Expected RM {expected:.2f} ({rate_label} of basic salary)"
        }
 
    # SOCSO: 0.5% of gross, capped at RM 5,000
    if fields.get("socso") and gross > 0:
        expected = _socso_lookup(gross)
        actual = fields["socso"]
        tolerance = max(expected * 0.15, 0.50)
        validations["socso"] = {
            "expected": round(expected, 2),
            "actual": round(actual, 2),
            "status": "✅ Correct" if abs(actual - expected) <= tolerance else "⚠️ Discrepancy",
            "note": f"Expected ~RM {expected:.2f} (0.5% of gross, capped RM 6,000)"
        }
 
    # EIS: 0.2% of gross, capped at RM 5,000
    if fields.get("eis") and gross > 0:
        expected = _eis_lookup(gross)
        actual = fields["eis"]
        tolerance = max(expected * 0.15, 0.30)
        validations["eis"] = {
            "expected": round(expected, 2),
            "actual": round(actual, 2),
            "status": "✅ Correct" if abs(actual - expected) <= tolerance else "⚠️ Discrepancy",
            "note": f"Expected ~RM {expected:.2f} (0.2% of gross, capped RM 6,000)"
        }
 
    # PCB: progressive tax estimate
    if fields.get("pcb") is not None and gross > 0:
        actual = fields["pcb"]
        estimated = _pcb_estimate((basic or gross) * 12) / 12
        if actual == 0:
            status = "✅ Zero PCB (low tax bracket or reliefs applied)"
        elif estimated <= 0:
            status = "✅ Low income — minimal PCB expected"
        else:
            ratio = actual / estimated if estimated > 0 else 1
            status = "✅ Within range" if 0.2 <= ratio <= 3.0 else "⚠️ Unusual — verify with HR"
        validations["pcb"] = {
            "actual": round(actual, 2),
            "estimated": round(estimated, 2),
            "status": status,
            "note": "PCB varies by personal reliefs, dependants, and tax bracket"
        }
 
    return validations
 
 
def _socso_lookup(gross: float) -> float:
    """SOCSO employee = 0.5% of gross, capped at RM 6,000"""
    return round(min(gross, 6000) * 0.005, 2)
 
 
def _eis_lookup(gross: float) -> float:
    """EIS employee = 0.2% of gross, capped at RM 6,000"""
    return round(min(gross, 6000) * 0.002, 2)
 
 
def _pcb_estimate(annual: float) -> float:
    """
    PCB based on official LHDN Malaysia progressive tax rates.
    Source: hasil.gov.my — no upper cap, max marginal rate 30%.
    """
    taxable = max(0, annual - 9000)
    if taxable <= 5000:    return 0
    if taxable <= 20000:   return (taxable - 5000) * 0.01
    if taxable <= 35000:   return 150 + (taxable - 20000) * 0.03
    if taxable <= 50000:   return 600 + (taxable - 35000) * 0.06
    if taxable <= 70000:   return 1500 + (taxable - 50000) * 0.11
    if taxable <= 100000:  return 3700 + (taxable - 70000) * 0.19
    if taxable <= 400000:  return 9400 + (taxable - 100000) * 0.25
    if taxable <= 600000:  return 84400 + (taxable - 400000) * 0.26
    if taxable <= 2000000: return 136400 + (taxable - 600000) * 0.28
    return 528400 + (taxable - 2000000) * 0.30
 
 
# ─────────────────────────────────────────
# STEP 4 — FINANCIAL CALCULATIONS
# ─────────────────────────────────────────
def calculate_financials(fields: dict) -> dict:
    net = fields.get("net_pay") or 0
    epf = fields.get("epf_employee") or 0
    epf_employer = fields.get("epf_employer") or round(epf * 13 / 11, 2)
    monthly_epf_total = epf + epf_employer

    rate = 0.055 / 12
    months = 30 * 12
    epf_projection = (
        monthly_epf_total * ((1 + rate) ** months - 1) / rate
        if rate > 0 else 0
    )

    max_monthly_installment = round(net * 0.60, 2)

    # Car loan: hire purchase flat rate 3.5%, 9 years
    car_rate_monthly = 0.035 / 12
    car_months = 9 * 12
    car_max_loan = max_monthly_installment * (1 - (1 + car_rate_monthly) ** -car_months) / car_rate_monthly

    # House loan: reducing balance 4.2%, 35 years
    house_rate_monthly = 0.042 / 12
    house_months = 35 * 12
    house_max_loan = max_monthly_installment * (1 - (1 + house_rate_monthly) ** -house_months) / house_rate_monthly

    return {
        "dsr_max_monthly_installment": max_monthly_installment,
        "car_max_loan":                round(car_max_loan, 2),
        "house_max_loan":              round(house_max_loan, 2),
        "budget_needs_50":             round(net * 0.50, 2),
        "budget_wants_30":             round(net * 0.30, 2),
        "budget_savings_20":           round(net * 0.20, 2),
        "epf_30yr_projection":         round(epf_projection, 2),
        "dsr_max_monthly_loan":        max_monthly_installment,  # backward compat
    }
 
 
# ─────────────────────────────────────────
# STEP 5 — CHARTS
# ─────────────────────────────────────────
def generate_charts(fields: dict, calculations: dict, output_dir: str):
    labels, sizes, colors = [], [], []
    for label, key, color in [
        ("EPF",     "epf_employee", "#2196F3"),
        ("SOCSO",   "socso",        "#4CAF50"),
        ("EIS",     "eis",          "#FF9800"),
        ("PCB Tax", "pcb",          "#F44336"),
    ]:
        v = fields.get(key)
        if v and v > 0:
            labels.append(label); sizes.append(v); colors.append(color)
 
    # Skip attendance penalties — only show statutory deductions
    skip_keywords = ["early out", "lateness", "time off", "late", "absent",
                     "lewat", "cuti", "overtime deduction"]
    for name, val in (fields.get("other_deductions") or {}).items():
        v = _to_float(val)
        if v and v > 0:
            if not any(kw in name.lower() for kw in skip_keywords):
                labels.append(name[:12]); sizes.append(v); colors.append("#607D8B")
 
    net = fields.get("net_pay", 0)
    if net and net > 0:
        labels.append("Take Home"); sizes.append(net); colors.append("#9C27B0")
 
    # Group tiny slices (< 1%) into "Others"
    total = sum(sizes)
    if total > 0:
        main_labels, main_sizes, main_colors = [], [], []
        others_total = 0
        for l, s, c in zip(labels, sizes, colors):
            if s / total < 0.01 and l != "Take Home":
                others_total += s
            else:
                main_labels.append(l)
                main_sizes.append(s)
                main_colors.append(c)
        if others_total > 0:
            main_labels.append("Others")
            main_sizes.append(others_total)
            main_colors.append("#455A64")
        labels, sizes, colors = main_labels, main_sizes, main_colors
 
    fig1, ax1 = plt.subplots(figsize=(7, 5.5))
    fig1.patch.set_facecolor("#1a1a2e")
    ax1.set_facecolor("#1a1a2e")
    if sizes:
        wedges, _, autotexts = ax1.pie(
            sizes, colors=colors[:len(sizes)],
            autopct=lambda p: f"{p:.1f}%" if p >= 2 else "",
            startangle=140,
            wedgeprops=dict(edgecolor="#1a1a2e", linewidth=2),
            textprops=dict(color="white", fontsize=10, fontweight="bold"))
        ax1.legend(wedges, [f"{l}  RM {s:,.2f}" for l, s in zip(labels, sizes)],
                   loc="lower center", bbox_to_anchor=(0.5, -0.20),
                   ncol=2, frameon=False, labelcolor="white", fontsize=9)
 
    gross = fields.get("gross_salary", sum(sizes) if sizes else 0) or 0
    ax1.set_title(f"Salary Breakdown\nGross: RM {gross:,.2f}",
                  color="white", fontsize=13, fontweight="bold", pad=15)
    pie_path = os.path.join(output_dir, "salary_chart.png")
    plt.tight_layout()
    plt.savefig(pie_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
 
    c = calculations
    bar_labels = ["Max Car\nLoan", "Max House\nLoan", "Needs\n(50%)", "Wants\n(30%)", "Savings\n(20%)"]
    bar_values = [c["car_max_loan"], c["house_max_loan"], c["budget_needs_50"], c["budget_wants_30"], c["budget_savings_20"]]
    bar_colors = ["#E91E63", "#9C27B0", "#2196F3", "#FF9800", "#4CAF50"]
    
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    fig2.patch.set_facecolor("#1a1a2e")
    ax2.set_facecolor("#1a1a2e")
    bars = ax2.bar(bar_labels, bar_values, color=bar_colors, edgecolor="#1a1a2e")
    for bar, val in zip(bars, bar_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                 f"RM {val:,.0f}", ha="center", va="bottom",
                 color="white", fontsize=9, fontweight="bold")
    ax2.set_title("Budget Allocation & Max Loan Eligibility",
                  color="white", fontsize=12, fontweight="bold")
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values():
        spine.set_color("#333355")
 
    loan_path = os.path.join(output_dir, "loan_chart.png")
    plt.tight_layout()
    plt.savefig(loan_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
 
    return pie_path, loan_path
 
 
# ─────────────────────────────────────────
# STEP 6 — INITIAL EXPLANATION
# ─────────────────────────────────────────
def generate_explanation(fields: dict, validations: dict, calculations: dict) -> str:
    summary = {
        "employee_name":    fields.get("employee_name"),
        "employer_name":    fields.get("employer_name"),
        "position":         fields.get("position"),
        "period":           fields.get("period"),
        "basic_salary":     fields.get("basic_salary"),
        "gross_salary":     fields.get("gross_salary"),
        "net_pay":          fields.get("net_pay"),
        "epf_employee":     fields.get("epf_employee"),
        "epf_employer":     fields.get("epf_employer"),
        "socso":            fields.get("socso"),
        "eis":              fields.get("eis"),
        "pcb":              fields.get("pcb"),
        "total_deduction":  fields.get("total_deduction"),
        "other_allowances": fields.get("other_allowances", {}),
        "other_deductions": fields.get("other_deductions", {}),
        "validations":      validations,
        "max_monthly_installment": calculations.get("dsr_max_monthly_installment"),
        "car_max_loan":            calculations.get("car_max_loan"),
        "house_max_loan":          calculations.get("house_max_loan"),
        "budget_needs_50":     calculations.get("budget_needs_50"),
        "budget_wants_30":     calculations.get("budget_wants_30"),
        "budget_savings_20":   calculations.get("budget_savings_20"),
        "epf_projection_30yr": calculations.get("epf_30yr_projection"),
    }
 
    prompt = f"""You are a friendly Malaysian financial advisor helping an employee understand their payslip.
 
Payslip data:
{json.dumps(summary, indent=2)}
 
Write a clear, friendly explanation in ENGLISH ONLY. No Malay words.
 
Structure:
1. **Salary Summary** — gross, deductions, net pay
2. **Deductions Explained** — EPF, SOCSO, EIS, PCB with ✅ or ⚠️ status
3. **Loan Eligibility** — max monthly installment (DSR 60%)
4. **50/30/20 Budget Tip** — practical breakdown
5. **Retirement Outlook** — EPF projection in 30 years
 
End with: "💬 Feel free to ask me anything! For example: 'Should I buy a Proton S70?' or 'Can I afford a house?'"
 
Keep it warm, under 300 words. English only. Use RM."""
 
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://payslip-analyzer.local",
        "X-Title": "Payslip Analyzer"
    }
 
    payload = {
        "model": EXPLANATION_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1500,
        "temperature": 0.3
    }
 
    response = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    raw = response.json()["choices"][0]["message"]["content"]
    return strip_thinking(raw)
 
 
# ─────────────────────────────────────────
# FINANCIAL ADVICE — Follow-up questions
# ─────────────────────────────────────────
def generate_financial_advice(user_question: str, fields: dict, calculations: dict) -> str:
    net = fields.get("net_pay") or 0
    gross = fields.get("gross_salary") or 0
    max_installment = calculations.get("dsr_max_monthly_installment") or calculations.get("dsr_max_monthly_loan") or 0
    savings = calculations.get("budget_savings_20") or 0

    prompt = f"""You are a concise Malaysian financial advisor. Answer directly and clearly.

Client salary data:
- Net Pay: RM {net:,.2f}
- Max monthly installment (DSR 60%): RM {max_installment:,.2f}/month
- Max car loan eligibility: RM {calculations.get('car_max_loan', 0):,.2f}  (9yr, 3.5% hire purchase)
- Max house loan eligibility: RM {calculations.get('house_max_loan', 0):,.2f}  (35yr, 4.2% reducing balance)
- Monthly savings budget (20%): RM {savings:,.2f}
- Monthly needs budget (50%): RM {calculations.get('budget_needs_50', 0):,.2f}

Known Malaysian car prices (2026):
- Proton S70 Executive: RM 89,900 | Premium: RM 97,900 | Flagship: RM 103,888
- Perodua Myvi: RM 50,000-60,000 | Perodua Ativa: RM 61,500-72,000
- Honda City: RM 80,000-100,000 | Proton X50: RM 79,200-103,800

Question: "{user_question}"

Reply in this exact format — no thinking, no working out, just the answer:

[One line: ✅ Yes / ⚠️ Possible but tight / ❌ Not recommended]

**Price & Installment**
- Est. price: RM X
- Monthly installment: RM X (use tenure/rate from user's question if specified; otherwise 9yr 3.5% for cars, 35yr 4.2% for houses)

**Your Numbers**
- DSR limit: RM {max_installment:,.2f}/month
- Installment fits? Yes / No

**Verdict**
2-3 sentences max. Be direct and honest.

**Tip**
One practical suggestion.

English only. No calculations shown. No thinking shown. Under 150 words total."""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://payslip-analyzer.local",
        "X-Title": "Payslip Analyzer"
    }

    payload = {
        "model": ADVICE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 700,
        "temperature": 0.7
    }

    response = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
 
 
# ─────────────────────────────────────────
# STEP 7 — FORMAT TELEGRAM MESSAGE
# ─────────────────────────────────────────
def format_telegram_message(fields, validations, calculations, explanation) -> str:
    gross  = fields.get("gross_salary") or 0
    basic  = fields.get("basic_salary") or 0
    net    = fields.get("net_pay") or 0
    name   = fields.get("employee_name") or "Employee"
    ic = fields.get("employee_ic")
    emp    = fields.get("employer_name") or ""
    period = fields.get("period") or "Not detected"
    pos    = fields.get("position") or ""
    age = _get_age_from_ic(ic) if ic else None
    # Clean null string values from AI
    if str(pos).lower() in ["null", "none", "not detected"]: pos = ""
    if str(emp).lower() in ["null", "none", "not detected"]: emp = ""
    if str(name).lower() in ["null", "none"]: name = "Employee"
 
    lines = [
        "📄 *Payslip Analysis*",
        f"👤 *{name}*" + (f" — {pos}" if pos else ""),
        f"🎂 Age: {age} years old" if age else "",
        f"🏢 {emp}" if emp and emp != "Not detected" else "",
        f"📅 {period}",
        "─────────────────────────────",
        f"💼 Basic Pay:     RM {basic:,.2f}" if basic else "",
        f"💰 Gross Salary:  RM {gross:,.2f}" if gross else "💰 Gross: Not detected",
        "",
        "*Deductions:*",
    ]

    epf_rate_label = "EPF (5.5%)" if (
        validations.get("epf", {}).get("note", "").find("5.5%") != -1
    ) else "EPF (11%)"

    for label, field_key, val_key in [
        (epf_rate_label, "epf_employee", "epf"),
        ("SOCSO",        "socso",        "socso"),
        ("EIS",          "eis",          "eis"),
        ("PCB Tax",      "pcb",          "pcb"),
    ]:
        val = fields.get(field_key)
        if val is not None and val > 0:
            status = ""
            if val_key and val_key in validations:
                status = "  " + validations[val_key]["status"]
            lines.append(f"  {label}: RM {val:,.2f}{status}")
 
    for k, v in (fields.get("other_deductions") or {}).items():
        fv = _to_float(v)
        if fv and fv > 0:
            lines.append(f"  {k}: RM {fv:,.2f}")
 
    total = fields.get("total_deduction")
    if total:
        lines.append("  ─────────────────")
        lines.append(f"  *Total Deductions: RM {total:,.2f}*")
 
    lines += [
        "",
        "─────────────────────────────",
        f"🏦 *Net Pay: RM {net:,.2f}*" if net else "🏦 Net Pay: Not detected",
        f"🚗 Max Car Loan: RM {calculations['car_max_loan']:,.2f}  (9yr, 3.5%)",
        f"🏠 Max House Loan: RM {calculations['house_max_loan']:,.2f}  (35yr, 4.2%)",
        "─────────────────────────────",
        "*💡 What does this mean?*",
        "",
        explanation,
        "",
        "─────────────────────────────",
        "_⚠️ Disclaimer: Estimates only. Verify with HR or a licensed financial advisor._",
    ]
 
    return "\n".join(l for l in lines if l is not None)
 
 
# ─────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────
def analyze_payslip(image_path: str, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    result = {"success": False, "error": None}
 
    try:
        print("⏳ Step 1: Preprocessing image...")
        preprocessed_path, blur_warning = preprocess_image(image_path)
 
        print("⏳ Step 2: AI extraction (llama-4-maverick on NVIDIA NIM)...")
        fields = extract_fields_with_ai(preprocessed_path)
        print(f"   Extracted fields:\n{json.dumps(fields, indent=2)}")
 
        print("⏳ Step 3: Validating deductions...")
        validations = validate_deductions(fields)
 
        print("⏳ Step 4: Calculating financials...")
        calculations = calculate_financials(fields)
 
        print("⏳ Step 5: Generating charts...")
        pie_path, loan_path = generate_charts(fields, calculations, output_dir)
 
        print("⏳ Step 6: Generating explanation...")
        explanation = generate_explanation(fields, validations, calculations)
 
        print("⏳ Step 7: Formatting message...")
        message = format_telegram_message(fields, validations, calculations, explanation)
 
        result = {
            "success": True,
            "message": message,
            "charts": [pie_path, loan_path],
            "fields": fields,
            "validations": validations,
            "calculations": calculations,
            "blur_warning": str(blur_warning),
        }
 
    except Exception as e:
        result["error"] = str(e)
        result["message"] = (
            f"❌ Sorry, I could not analyse this payslip.\n"
            f"Error: {e}\n\n"
            f"Please make sure the image is clear and well-lit."
        )
 
    result_path = os.path.join(output_dir, "payslip_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
 
    print(f"\n✅ Done! Results saved to {result_path}")
    return result
 
 
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python payslip_analyzer.py <image_path> <output_dir>")
        sys.exit(1)
    result = analyze_payslip(sys.argv[1], sys.argv[2])
    if result["success"]:
        print("\n─── TELEGRAM MESSAGE PREVIEW ───")
        print(result["message"])
    else:
        print(f"\n❌ Failed: {result['error']}")
        sys.exit(1)