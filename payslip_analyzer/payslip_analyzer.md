# Skill: Payslip Analyzer

## Description
Analyses Malaysian payslip photos sent by users in Telegram. Extracts salary fields using OCR, validates EPF/SOCSO/PCB/EIS deductions against official Malaysian rates, calculates separate car and house loan eligibility, generates charts, and returns a plain-English summary. Also answers follow-up financial questions based on the user's salary.

## Commands
| Command | Description |
|---|---|
| `/payslip` | Start payslip analysis — send this before your payslip photo |
| `/reset` | Clear your session and start over |
| `/help` | Show help message |

## Trigger
Activate this skill when:
- The user sends `/payslip` command
- The user sends a **photo or image** after `/payslip`
- The user says: "payslip", "gaji", "salary", "analyze my payslip"
- The user asks follow-up financial questions after analysis:
  - "Should I buy a Proton S70?"
  - "Can I afford a RM 200k car with 5 year loan?"
  - "Can I afford a house in KL?"
  - "How much should I save?"

## Conversation Flow

```
User: /payslip
Hermes: 📸 Please send me a photo of your payslip!

User: [sends payslip photo]
Hermes: ⏳ Analysing your payslip...
Hermes: [sends analysis + charts]
Hermes: 💬 Feel free to ask me anything about your salary!

User: Should I buy a Proton S70?
Hermes: [sends financial advice]

User: Can I afford a RM 200k car with 5 year loan?
Hermes: [sends financial advice using user-specified tenure]
```

## Execution

When user sends `/payslip`:
Reply with: "📸 Please send me a clear photo of your payslip! Make sure all text is visible."

When user sends a photo after `/payslip`, run this command:

```bash
cd /Users/ziyuan/Documents/vscode/hermes_payslip/payslip_analyzer && \
source /Users/ziyuan/Documents/vscode/hermes_payslip/.venv/bin/activate && \
export $(grep -v '^#' .env | xargs) && \
python payslip_analyzer.py {image_path} /Users/ziyuan/Documents/vscode/hermes_payslip/payslip_analyzer/output
```

Then read the result from:
```
/Users/ziyuan/Documents/vscode/hermes_payslip/payslip_analyzer/output/payslip_result.json
```

Send back to user:
1. The `message` field from `payslip_result.json`
2. `salary_chart.png` as a photo
3. `loan_chart.png` as a photo
4. "💬 You can now ask me anything! Try: Should I buy a Proton S70? or Can I afford a house in KL?"

## What This Skill Does (Step by Step)

1. **Receive** `/payslip` command from user
2. **Prompt** user to send payslip photo
3. **Receive** payslip photo from Telegram
4. **Save** image to a temporary path
5. **Run** `payslip_analyzer.py` with the image path
6. **Preprocess** image — grayscale, CLAHE contrast enhancement, deskew using Hough Line Transform
7. **OCR** — send to NVIDIA NIM llama-4-maverick to extract all salary fields as JSON
8. **Validate** EPF (11% below 60, 5.5% age 60+), SOCSO (0.5%), EIS (0.2%) against official Malaysian rates
9. **Calculate** DSR loan eligibility (60% of net pay), separate car and house loan amounts, 50/30/20 budget, EPF 30-year projection
10. **Generate** salary breakdown pie chart and budget/loan bar chart using matplotlib
11. **Explain** results in plain English using OpenRouter (nemotron-3-super-120b)
12. **Send** text summary + charts back to user in Telegram
13. **Remember** salary data for follow-up financial questions

## Environment Variables Required
```
NIM_API_KEY=<your NVIDIA NIM API key>
OPENROUTER_API_KEY=<your OpenRouter API key>
TELEGRAM_BOT_TOKEN=<your Telegram bot token>
```

## Tech Stack
| Component | Model/Tool | Platform |
|---|---|---|
| OCR | `meta/llama-4-maverick-17b-128e-instruct` | NVIDIA NIM (Free) |
| Explanation | `nvidia/nemotron-3-super-120b-a12b:free` | OpenRouter (Free) |
| Financial Advice | `openai/gpt-oss-120b:free` | OpenRouter (Free) |
| Image Processing | OpenCV + Pillow | Local Python |
| Charts | matplotlib | Local Python |
| Interface | Telegram Bot | python-telegram-bot |

## Malaysian Statutory Rates Reference
| Contribution | Employee Rate | Cap |
|---|---|---|
| EPF (KWSP) | 11% (below 60) / 5.5% (age 60+) of basic | No cap |
| SOCSO (PERKESO) | 0.5% of gross | RM 6,000 gross |
| EIS (SIP) | 0.2% of gross | RM 6,000 gross |
| PCB (Tax) | Progressive 0-30% | No cap |

## Loan Eligibility Calculation
| Loan Type | Tenure | Interest Rate | Method |
|---|---|---|---|
| Car Loan | 9 years | 3.5% flat (hire purchase) | Reducing balance formula |
| House Loan | 35 years | 4.2% p.a. | Reducing balance formula |
| DSR Cap | — | — | 60% of net pay (max monthly installment) |

If the user specifies a custom tenure or rate (e.g. "5 year loan", "RM 200k car"), the advice engine uses the user's values instead of the defaults.

## Output Format
Send back in Telegram in this order:
1. Payslip analysis text (Markdown)
2. `salary_chart.png` — pie chart of salary breakdown
3. `loan_chart.png` — bar chart showing Max Car Loan, Max House Loan, Needs (50%), Wants (30%), Savings (20%)
4. Follow-up prompt inviting financial questions

## Error Handling
- Blurry image → warn user to retake photo
- 429 rate limit → retry automatically up to 3 times
- Missing fields → mark as "Not detected", continue pipeline
- Timeout → increase to 90 seconds

## Privacy Rules
- Payslip image deleted immediately after processing
- No personal data stored on any server
- Session data cleared on /reset or bot restart
- Always include disclaimer: results are estimates only

## Example Interaction

**User:** `/payslip`

**Hermes:**
> 📸 Please send me a clear photo of your payslip!

**User:** [sends payslip photo]

**Hermes:**
> ⏳ Analysing your payslip...

**Hermes:** *(after processing)*
> 📄 Payslip Analysis — Jeffrey Toh
> 📅 January 2019
> 💼 Basic Pay: RM 3,000.00
> 💰 Gross Salary: RM 3,947.05
>
> Deductions:
>   EPF (11%): RM 319.00 ✅ Correct
>   SOCSO: RM 19.25 ✅ Correct
>   EIS: RM 7.70 ✅ Correct
>   PCB Tax: RM 12.40 ✅ Within range
>
> 🏦 Net Pay: RM 3,486.61
> 🚗 Max Car Loan: RM 175,432.00  (9yr, 3.5%)
> 🏠 Max House Loan: RM 431,205.00  (35yr, 4.2%)
>
> [salary_chart.png]
> [loan_chart.png]
>
> 💬 Feel free to ask me anything!

**User:** Should I buy a Proton S70 Flagship?

**Hermes:**
> ⚠️ Possible but tight
> Price: RM 103,888
> Monthly installment: RM 1,023 (9yr, 3.5%)
> DSR limit: RM 2,091.97/month — fits, but leaves little room.
> Tip: Consider the S70 Executive at RM 89,900 instead.

**User:** Can I afford a RM 200k car with 5 year loan?

**Hermes:**
> ❌ Not recommended
> Price: RM 200,000
> Monthly installment: RM 3,467 (5yr, 3.5%)
> DSR limit: RM 2,091.97/month — exceeds your limit.
> Tip: Extend to 9 years or choose a car under RM 175,000.

## Notes
- Car prices are maintained in `payslip_analyzer.py` — update there, not here
- Do not provide tax filing advice
- Always recommend verifying with HR or licensed advisor
- Use dummy payslips during testing only
- Total cost: RM 0 (all free APIs)