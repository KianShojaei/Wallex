# Wallex Agent — موتور معاملاتی Agentic AI برای چالش والکس (هکاتون StarCoach)

> **خلاصه یک‌خطی:** یک ایجنت معاملاتی خودکار، Long-only و Spot، روی بازارهای تومانی والکس که با یک لایه‌ی سیگنال قطعی (Mean-Reversion با فیلتر روند) فرصت پیدا می‌کند، یک تحلیل‌گر LLM آن را تأیید/رد/کوچک می‌کند، و یک موتور ریسک سخت‌گیر اندازه، حد ضرر و تسویه را کنترل می‌کند. هر تصمیم (حتی «کاری نکن») با دلیل در لاگ JSONL ثبت می‌شود و کل اجرا قابل تکرار است.

این مستند مطابق «موارد تحویلی» بریف چالش نوشته شده: معماری و منطق، نحوه استفاده از AI و ابزارهای والکس، شرایط خرید/فروش، مدیریت ریسک، و نحوه اجرای مجدد و ارزیابی.

---

## ۱. معماری

```
                 ┌──────────────────────────────── هر ۶۰ ثانیه ────────────────────────────────┐
 Wallex REST ───►│ MarketData ──► Features ──► Signals (قطعی) ──► AI Analyst ──► RiskManager ──► Broker │──► Wallex REST
 /v1/markets     │  کندل‌ها       RSI/BB/EMA     score 0..1        approve /      سایز، SL/TP     live /   │   POST /v1/account/orders
 /v1/udf/history │  اردربوک       ATR/vol-z      + reasons         reject /       drawdown        shadow / │
 /v1/depth       │                               + vetoes          resize         liquidation     backtest │
                 └───────────────────────────────┬────────────────────────────────────────────────────────┘
                                                 ▼
                                     Decision Log (JSONL)  +  Portfolio State (JSON)
```

| ماژول | مسیر | نقش |
|---|---|---|
| کلاینت والکس | `src/wallex_agent/wallex/client.py` | REST با retry/backoff و rate-limit؛ هدر `X-API-Key` |
| پل MCP والکس | `src/wallex_agent/wallex/mcp_tools.py` | اتصال اختیاری به `https://api.wallex.ir/mcp` (Streamable HTTP) و تبدیل tool‌های آن به فرمت function-calling برای تحلیل‌گر |
| داده | `src/wallex_agent/data/market_data.py` | کندل UDF → DataFrame، کش CSV برای بک‌تست |
| ویژگی‌ها | `src/wallex_agent/data/features.py` | RSI، Bollinger، EMA(20/50/100)، ATR، z-score حجم، عدم‌تعادل اردربوک |
| سیگنال قطعی | `src/wallex_agent/strategy/signals.py` | امتیاز مرکب ۰..۱ + دلایل + وتوها |
| تحلیل‌گر AI | `src/wallex_agent/ai/analyst.py`, `ai/llm.py` | دو نقش: رژیم بازار (ساعتی) و تصمیم روی هر کاندید؛ خروجی JSON سخت با pydantic؛ کش و حالت replay |
| ریسک | `src/wallex_agent/risk/manager.py` | تنها جایی که سایز، SL/TP، توقف‌ها و تسویه تعیین می‌شود |
| اجرا | `src/wallex_agent/execution/broker.py` | `LiveBroker` (سفارش واقعی)، `SimulatedBroker` (شدو/بک‌تست) با یک اینترفیس |
| هسته تصمیم | `src/wallex_agent/engine/core.py` | `Agent.step()` که در live/shadow/backtest **دقیقاً یکی** است |
| حلقه اجرا | `src/wallex_agent/engine/loop.py` | وایرینگ، resume بعد از کرش، سیگنال‌های خاتمه |
| بک‌تست | `src/wallex_agent/backtest/runner.py` | رویدادمحور، همان هسته، متریک‌ها و منحنی equity |
| لاگ تصمیم | `src/wallex_agent/decision_log/logger.py` | JSONL append-only |
| CLI | `src/wallex_agent/cli.py` | `check / fetch-data / backtest / run / status / liquidate / report` |

---

## ۲. منطق استراتژی: «AI-gated Mean Reversion با فیلتر روند»

**چرا این استراتژی؟** افق ۳ روزه، سرمایه ۵ میلیون تومان، کارمزد حدود ۰.۲۵٪ هر طرف، و معیار داوری «بازدهی خالص + Drawdown کم». بازگشت به میانگین روی کندل‌های ساعتی، با حد ضرر تنگ و فیلتر روند، فرکانس معامله‌ی معقول (چند معامله در روز) و ریسک محدود می‌دهد، و منطق آن کاملاً قابل توضیح است.

### ۲.۱ شرایط ورود (خرید) — لایه‌ی قطعی
همه‌ی محاسبات روی کندل‌های بسته‌شده (`data.resolution`، پیش‌فرض ۱ ساعته):

| شرط | معیار | نقش در امتیاز |
|---|---|---|
| اشباع فروش | `RSI(14) ≤ 32` | ۴۵٪ (RSI پایین‌تر → امتیاز بیشتر) |
| کشش آماری | قیمت در/زیر باند پایین بولینگر (`bb_pos ≤ 0.15`) | ۳۵٪ |
| فیلتر روند | امتیاز ۱ اگر قیمت بالای EMA100، ۰.۶ اگر EMA20 ≥ EMA50، وگرنه ۰.۳ | ۱۰٪ |
| تأیید اردربوک | عدم تعادل حجم ۱۰ سطح اول (bid سنگین → بهتر) | ۱۰٪ |
| جریمه نوسان | اگر ATR% > 1.5 امتیاز کم می‌شود | − |

**وتوها (رد قطعی):** روند نزولی قوی (قیمت < EMA100 و EMA20 < EMA50 و بازده ۲۴h < −۴٪)، بازار مرده (z-score حجم < −۱)، اسپرد > ۰.۳۵٪، نبود شرط اشباع فروش.

فقط کاندیدهایی با `score ≥ 0.55` به لایه‌ی AI می‌رسند.

### ۲.۲ شرایط خروج (فروش) — موتور ریسک
به ترتیب اولویت، در هر حلقه برای هر پوزیشن باز:
1. **تسویه پایان چالش:** ۴۵ دقیقه قبل از پایان رسمی، همه چیز به تومان (قانون «سود باید realized و تومانی باشد»).
2. **Kill switch** (Drawdown کل ≥ ۶٪): همه چیز بسته و ایجنت متوقف.
3. **حد ضرر** ۱.۰٪ زیر قیمت ورود (AI فقط می‌تواند تنگ‌تر کند).
4. **حد سود** ۱.۶٪ (AI می‌تواند در بازه ۰.۵٪ تا ۲.۴٪ تنظیم کند).
5. **Trailing stop** ۰.۸٪ از سقف، پس از رسیدن به نصف حد سود.
6. **Time stop** ۱۲ ساعت.
7. **بازگشت به میانگین:** RSI ≥ 55 و قیمت بالای وسط باند، فقط اگر سود از دوبرابر کارمزد بیشتر باشد.

---

## ۳. استفاده واقعی از AI (نه فقط اسم)

AI در دو نقطه‌ی تصمیم واقعی قرار دارد و خروجی‌اش **مستقیماً** روی ورود/اندازه اثر می‌گذارد، اما همیشه داخل حدود موتور ریسک:

| نقش | مدل | زمان فراخوانی | خروجی (JSON سخت) | اثر |
|---|---|---|---|---|
| **تحلیل رژیم بازار** | `openai/gpt-6-sol` | هر ۶۰ دقیقه برای هر نماد که کاندید دارد | `regime, confidence, allow_entries, size_factor, summary` | `allow_entries=false` → هیچ ورودی؛ `size_factor` سایز را کوچک می‌کند |
| **قضاوت روی کاندید** | `openai/gpt-6-luna` | برای هر کاندید با score ≥ 0.55 | `action (approve/approve_reduced/reject), confidence, size_factor, stop_loss_pct, take_profit_pct, rationale, key_risks` | رد → ورود نمی‌شود؛ تأیید با confidence < 0.60 → رد خودکار؛ `rationale` در لاگ تصمیم ذخیره می‌شود |

**قیود طراحی:**
- AI فقط می‌تواند ریسک را **کم** کند (size_factor ∈ [0,1]، استاپ فقط تنگ‌تر). موتور ریسک همه‌ی پیشنهادها را clamp می‌کند (`risk/manager.py::plan_entry`).
- AI هیچ‌وقت مستقیماً سفارش نمی‌زند؛ tool‌های MCP مربوط به سفارش/برداشت هرگز به LLM داده نمی‌شوند (`mcp_tools.py::as_openai_tools`).
- خروجی با pydantic اعتبارسنجی می‌شود؛ خروجی نامعتبر یک بار اصلاح و در غیر این صورت رد می‌شود.
- اگر LLM در دسترس نباشد، سیاست محافظه‌کار قطعی اجرا می‌شود (فقط score ≥ 0.75 با نصف سایز) و لاگ آن را با `ai_unavailable` علامت می‌زند.
- **تکرارپذیری:** همه‌ی فراخوانی‌ها (پرامپت، مدل، خروجی خام، هزینه، تأخیر) در `logs/ai_cache/<hash>.json` ذخیره می‌شوند؛ با `--ai replay` همان پاسخ‌ها بایت‌به‌بایت بازپخش می‌شوند و داور می‌تواند اجرای ما را بدون هزینه و بدون تصادفی‌بودن مدل بازسازی کند.

نمونه‌ی یک تصمیم واقعی مدل روی یک کاندید (داده‌ی مصنوعی تست):
```json
{"action":"reject","confidence":0.78,"size_factor":0.0,
 "rationale":"RSI 24.5 and bb_pos -0.28 show oversold conditions, but momentum is strongly negative: ret_1 -2.15%, ret_6 -2.83%, ret_24 -3.73%. Price is about 2.6% below both fast and slow EMAs ... A 1.6% target leaves only about 0.9% after round-trip fees and spread, without a reversal signal to support it.",
 "key_risks":["Continuing downside despite oversold readings","Unknown market regime","Fees and spread materially reduce the target's net return"]}
```

### ابزارهای والکس
- **Wallex REST API** (`https://api.wallex.ir`): مسیر اصلی داده و اجرا (`/v1/markets`, `/v1/depth`, `/v1/udf/history`, `/v1/account/balances`, `/v1/account/fee`, `/v1/account/orders`, `/v1/account/openOrders`).
- **Wallex MCP** (`https://api.wallex.ir/mcp`): با فلگ `--mcp`، ابزارهای فقط‌خواندنی سرور MCP والکس به تحلیل‌گر داده می‌شوند تا در تحلیل رژیم، کانتکست بیشتر (آمار بازار، اردربوک، معاملات اخیر) بگیرد. اجرای سفارش همیشه از REST و زیر نظر موتور ریسک است.

---

## ۴. مدیریت سرمایه و ریسک (`config/settings.yaml` → `risk:`)

| قاعده | مقدار پیش‌فرض |
|---|---|
| سقف هر پوزیشن | ۲۰٪ equity |
| سقف کل در بازار (هرگز با کل پول وارد نمی‌شود) | ۴۰٪ equity |
| حداکثر پوزیشن باز | ۲ |
| حد ضرر / حد سود / Trailing | ۱.۰٪ / ۱.۶٪ / ۰.۸٪ |
| Time stop | ۱۲ ساعت |
| توقف روزانه (بدون ورود جدید تا فردا) | Drawdown روزانه ≥ ۳٪ |
| Kill switch (بستن همه و توقف) | Drawdown کل ≥ ۶٪ |
| Cooldown هر نماد بعد از ضرر | ۹۰ دقیقه |
| حداقل اندازه سفارش | ۳۰۰٬۰۰۰ تومان |
| مقیاس نوسان | سایز × min(1, 1/ATR%) |
| تسویه پایان چالش | ۴۵ دقیقه قبل از پایان، همه به TMN |

اجرای سفارش: LIMIT قابل‌اجرا (best ask + ۰.۱٪) با لغو خودکار بعد از ۴۵ ثانیه در صورت پر نشدن؛ یا MARKET با تنظیم `execution.order_type`.

---

## ۵. اجرای مجدد و ارزیابی (برای داوران)

### نصب
```bash
git clone <this repo> && cd Wallex
python -m venv .venv && source .venv/bin/activate     # Python 3.11+
pip install -r requirements.txt && pip install -e .
cp .env.example .env        # WALLEX_API_KEY و AI_API_KEY را پر کنید (هرگز کامیت نشود)
```

### تست اتصال
```bash
wallex-agent check          # API عمومی والکس، حساب (اگر کلید باشد)، گیت‌وی AI
```

### بک‌تست
```bash
wallex-agent fetch-data --days 30                 # کندل‌های واقعی والکس → data/candles/*.csv
wallex-agent backtest --ai off                    # خط پایه‌ی کاملاً قطعی (فقط قواعد)
wallex-agent backtest --ai live                   # با AI (هزینه‌ی چند سنت)؛ پاسخ‌ها کش می‌شوند
wallex-agent backtest --ai replay                 # بازپخش دقیق همان اجرای AI
# بدون دسترسی به صرافی (داده‌ی مصنوعی فقط برای تست پایپ‌لاین):
python scripts/make_synthetic_data.py && wallex-agent backtest --ai off
```
خروجی هر بک‌تست در `logs/backtests/<run>/`: `metrics.json` (بازدهی خالص، Max DD، شارپ، تعداد معامله، win rate، profit factor، کارمزدها، دلایل خروج)، `equity.csv`، `trades.csv`، و `backtest_decisions.jsonl` (همان لاگ تصمیم).

### Shadow Testing (بازار واقعی، بدون سفارش)
```bash
wallex-agent run --mode shadow            # یا: docker compose up -d   (RUN_MODE=shadow)
wallex-agent status --mode shadow
wallex-agent report                       # خلاصه لاگ‌های تصمیم
```

### اجرای واقعی (حساب چالش)
```bash
wallex-agent run --mode live              # تأیید تایپی LIVE می‌خواهد؛ با -y در داکر
wallex-agent liquidate --mode live        # خروج اضطراری الگوریتمی به تومان
```
وضعیت پرتفوی در `state/portfolio_<mode>.json` ذخیره می‌شود؛ بعد از کرش/ری‌استارت ادامه می‌دهد (`docker compose` با `restart: unless-stopped`).

### تست‌های واحد
```bash
pytest -q        # ۱۲ تست: اندیکاتورها، سیگنال‌ها، ریسک (سایز/سقف‌ها/خروج‌ها/Drawdown)، بک‌تست end-to-end، تکرارپذیری، فالبک AI
```

---

## ۶. لاگ تصمیم (تحویلی)

`logs/decisions/<mode>_<run>.jsonl`، یک رکورد در هر خط. انواع رکورد:

| kind | معنی |
|---|---|
| `run_start` / `run_stop` | شروع/پایان با کانفیگ کامل (بدون کلیدها) |
| `scan` | نمادی بررسی شد و کاندید نشد (score، وتوها، RSI، bb_pos) |
| `regime` | خروجی تحلیل رژیم AI برای یک نماد |
| `reject` | کاندید رد شد؛ `stage` = `regime` / `ai` / `risk` با دلیل کامل |
| `no_entry` | گیت‌های سراسری ورود را بستند (drawdown، سقف پوزیشن، پنجره‌ی تسویه) |
| `entry` | **چه سیگنالی دید و چرا خرید:** سیگنال و دلایل، رژیم، تصمیم AI و `why`، طرح ریسک (سایز/SL/TP)، fill واقعی |
| `exit` | چرا فروخت (`reason`)، قیمت ورود/خروج، PnL تومانی و درصدی، مدت نگهداری، ارجاع به `entry_decision_id` |
| `kill_switch` / `error` | رخدادهای ایمنی |

چون هر سفارش فقط از `Agent.step()` عبور می‌کند و همان‌جا لاگ می‌شود، هر معامله‌ای در حساب که رکورد `entry`/`exit` متناظر نداشته باشد، به‌وضوح غیرالگوریتمی است (ما چنین معامله‌ای نداریم).

---

## ۷. انطباق با قوانین چالش

- فقط بازارهای `markets.allowed` معامله می‌شوند (فهرست رسمی را در همین کلید قرار دهید).
- فقط Spot و Long؛ هیچ معامله‌ی دستی، self-trading یا wash-trading: یک حساب، یک ایجنت، یک لاگ.
- نتیجه‌ی نهایی به‌صورت **realized در تومان**: تسویه‌ی خودکار قبل از پایان و `liquidate` اضطراری.
- کد تحویلی همان کدی است که معامله می‌کند (هش کامیت در `run_start` → ثبت می‌شود).

## ۸. ساختار پوشه‌ها
```
config/settings.yaml      همه‌ی پارامترها (استراتژی، ریسک، بازارها، زمان چالش)
src/wallex_agent/         کد منبع (جدول بخش ۱)
tests/                    تست‌های واحد و end-to-end
scripts/                  ابزارهای کمکی (داده‌ی مصنوعی)
logs/decisions/           لاگ تصمیم (تحویلی)      logs/ai_cache/ کش پاسخ‌های AI (برای replay)
logs/backtests/           خروجی بک‌تست‌ها          state/        وضعیت پرتفوی
data/candles/             کش کندل‌ها
```
