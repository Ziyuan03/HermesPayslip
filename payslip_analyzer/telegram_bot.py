"""
telegram_bot.py — Payslip Analyzer Telegram Bot
"""

import os, logging, tempfile
from dotenv import load_dotenv
load_dotenv() 
from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode, ChatAction
from payslip_analyzer import analyze_payslip, generate_financial_advice

logging.basicConfig(format="%(asctime)s — %(levelname)s — %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
NIM_API_KEY        = os.environ.get("NIM_API_KEY", "")
OUTPUT_DIR         = os.path.join(os.path.dirname(__file__), "output")
user_sessions      = {}


async def safe_delete(bot, chat_id, message_id):
    try:
        await bot.delete_message(chat_id=chat_id, message_id=message_id)
    except Exception:
        pass


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 *Welcome to Payslip Analyzer!*\n\n"
        "Send me a photo of your Malaysian payslip and I will:\n"
        "✅ Verify EPF, SOCSO, EIS and PCB deductions\n"
        "💰 Calculate your maximum loan eligibility\n"
        "📊 Show a salary breakdown chart\n"
        "💡 Explain everything in plain English\n"
        "🚗 Answer financial questions like:\n"
        "   Should I buy a Proton S70?\n"
        "   Can I afford a house in KL?\n\n"
        "📸 *Start by sending me a photo of your payslip!*",
        parse_mode=ParseMode.MARKDOWN
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 *How to use:*\n\n"
        "1. Send a clear photo of your payslip\n"
        "2. Wait about 15 seconds\n"
        "3. Receive your full breakdown and charts\n"
        "4. Ask any financial questions!\n\n"
        "*Example questions:*\n"
        "• Should I buy a Proton S70 Flagship?\n"
        "• Can I afford a RM 400,000 house?\n"
        "• How much should I save each month?\n\n"
        "/reset — Clear your session\n"
        "/privacy — Data policy",
        parse_mode=ParseMode.MARKDOWN
    )


async def privacy_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🔒 *Privacy Policy*\n\n"
        "• Photo deleted immediately after processing\n"
        "• Salary data only in memory for your session\n"
        "• Nothing stored on any server\n"
        "• Results are estimates — verify with HR",
        parse_mode=ParseMode.MARKDOWN
    )


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id in user_sessions:
        del user_sessions[update.effective_user.id]
    await update.message.reply_text("🔄 Session cleared! Send a new payslip photo to start again.")

async def payslip_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📸 Please send me a clear photo of your payslip!\n\n"
        "Make sure:\n"
        "• Good lighting\n"
        "• All text is visible\n"
        "• Payslip is flat, no shadows"
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user    = update.effective_user
    logger.info(f"Photo from {user.id} ({user.first_name})")

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    wait_msg = await update.message.reply_text("📸 Got your payslip! Analysing now...\n⏳ About 15 seconds...")

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            photo = update.message.photo[-1]
            file  = await context.bot.get_file(photo.file_id)
            image_path = os.path.join(tmpdir, "payslip.jpg")
            await file.download_to_drive(image_path)

            await context.bot.edit_message_text(
                chat_id=chat_id, message_id=wait_msg.message_id,
                text="🔍 Reading payslip with AI...\n⏳ Almost done..."
            )

            os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
            os.environ["NIM_API_KEY"]         = NIM_API_KEY

            result = analyze_payslip(image_path, OUTPUT_DIR)
            await safe_delete(context.bot, chat_id, wait_msg.message_id)

            if result["success"]:
                user_sessions[user.id] = {
                    "fields":       result["fields"],
                    "calculations": result["calculations"],
                }

                # Send text — always fallback to plain text if markdown fails
                try:
                    await update.message.reply_text(result["message"], parse_mode=ParseMode.MARKDOWN)
                except Exception:
                    await update.message.reply_text(result["message"])

                # Send charts
                for chart_path in result.get("charts", []):
                    if os.path.exists(chart_path):
                        with open(chart_path, "rb") as f:
                            await context.bot.send_photo(chat_id=chat_id, photo=InputFile(f))

                if result.get("blur_warning") == "True":
                    await update.message.reply_text(
                        "⚠️ Your image looks blurry — results may be less accurate. Try a clearer photo.")

                await update.message.reply_text(
                    "💬 You can now ask me anything based on your salary!\n\n"
                    "Try:\n"
                    "• Should I buy a Proton S70 Flagship?\n"
                    "• Can I afford a RM 400k house?\n"
                    "• How much should I save monthly?"
                )
            else:
                await update.message.reply_text(
                    f"❌ Analysis failed\n\n{result.get('message', 'Unknown error')}\n\nPlease try again with a clearer photo.")

        except Exception as e:
            logger.error(f"Photo error: {e}", exc_info=True)
            await safe_delete(context.bot, chat_id, wait_msg.message_id)
            await update.message.reply_text(f"❌ Analysis failed: {str(e)[:200]}\n\nPlease try again.")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user    = update.effective_user
    chat_id = update.effective_chat.id
    text    = update.message.text.strip()

    if user.id in user_sessions:
        session      = user_sessions[user.id]
        fields       = session["fields"]
        calculations = session["calculations"]

        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        wait_msg = await update.message.reply_text("💭 Thinking...")

        try:
            os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
            os.environ["NIM_API_KEY"]         = NIM_API_KEY
            advice = generate_financial_advice(text, fields, calculations)
            await safe_delete(context.bot, chat_id, wait_msg.message_id)
            try:
                await update.message.reply_text(f"💡 *Financial Advice*\n\n{advice}", parse_mode=ParseMode.MARKDOWN)
            except Exception:
                await update.message.reply_text(f"💡 Financial Advice\n\n{advice}")

        except Exception as e:
            logger.error(f"Advice error: {e}", exc_info=True)
            await safe_delete(context.bot, chat_id, wait_msg.message_id)
            await update.message.reply_text(f"❌ Advice failed: {str(e)[:200]}")
    else:
        await update.message.reply_text(
            "📸 Please send me a photo of your payslip first!\n\n"
            "Once I analyse it, you can ask me any financial questions.")


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📄 Please send a photo of your payslip, not a file. Take a screenshot and send that instead!")


def main():
    if not TELEGRAM_BOT_TOKEN:
        print("❌ ERROR: TELEGRAM_BOT_TOKEN not set!"); return
    if not OPENROUTER_API_KEY:
        print("❌ ERROR: OPENROUTER_API_KEY not set!"); return
    if not NIM_API_KEY:
        print("❌ ERROR: NIM_API_KEY not set!"); return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("🚀 Starting Payslip Analyzer Bot...")
    print("   OCR:         llama-4-maverick     (NVIDIA NIM)")
    print("   Explanation: nemotron-super-120b  (OpenRouter)")
    print("   Advice:      gpt-oss-120b         (OpenRouter)")
    print("   Press Ctrl+C to stop\n")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start",   start))
    app.add_handler(CommandHandler("help",    help_command))
    app.add_handler(CommandHandler("privacy", privacy_command))
    app.add_handler(CommandHandler("reset",   reset_command))
    app.add_handler(CommandHandler("payslip", payslip_command))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    print("✅ Bot is running!\n")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()


    