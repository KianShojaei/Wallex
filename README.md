# Wallex Agent — موتور معاملاتی Agentic AI برای چالش والکس (هکاتون StarCoach)

> **خلاصه یک‌خطی:** یک ایجنت معاملاتی خودکار، Long-only و Spot، روی بازارهای تومانی والکس. یک لایه‌ی قطعی «رژیم ارزی» تشخیص می‌دهد تومان در حال ضعیف شدن است یا نه و پیشنهاد می‌دهد چه سهمی از سرمایه در USDT باشد؛ یک تحلیل‌گر LLM سلامت رژیم را می‌سنجد و اندازه را تأیید/کوچک/رد می‌کند؛ و یک موتور ریسک سخت‌گیر سقف‌ها، حد ضرر، افزایش تدریجی و تسویه‌ی نهایی به تومان را کنترل می‌کند. دو ماژول دیگر (روندگرای کریپتو و بازگشت به میانگین) پیاده‌سازی شده‌اند اما چون روی داده‌ی واقعی بعد از کارمزد منفی بودند، پیش‌فرض خاموش‌اند. هر تصمیم (حتی «کاری نکن») با دلیل در لاگ JSONL ثبت می‌شود و کل اجرا قابل تکرار است.
>
> **چرا این طراحی؟** تحلیل ۳۰ روز کندل ۱ دقیقه‌ای واقعی والکس (`docs/research.md`) نشان داد همه‌ی ارزها به دلار منفی بوده‌اند و تمام سود بازارهای تومانی از افت تومان (USDT/TMN +۱۸٪) آمده؛ و با کارمزد ۰.۶٪ هر رفت‌وبرگشت، هیچ قاعده‌ی زمان‌بندی تکنیکال (۰ از ۶۴۸ ترکیب بازگشت به میانگین) خارج از نمونه مثبت نبود.

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
| ماژول‌های استراتژی | `src/wallex_agent/strategy/{base,registry,carry,trend,mean_reversion}.py` | هر ماژول کاندید (Signal با امتیاز، دلایل، وتوها و سقف/استاپ خودش) و «اشاره‌ی خروج» می‌دهد؛ `carry` هسته است |
| تحلیل‌گر AI | `src/wallex_agent/ai/analyst.py`, `ai/llm.py` | دو نقش: رژیم بازار (ساعتی) و تصمیم روی هر کاندید؛ خروجی JSON سخت با pydantic؛ کش و حالت replay |
| ریسک | `src/wallex_agent/risk/manager.py` | تنها جایی که سایز، SL/TP، توقف‌ها و تسویه تعیین می‌شود |
| اجرا | `src/wallex_agent/execution/broker.py` | `LiveBroker` (سفارش واقعی)، `SimulatedBroker` (شدو/بک‌تست) با یک اینترفیس |
| هسته تصمیم | `src/wallex_agent/engine/core.py` | `Agent.step()` که در live/shadow/backtest **دقیقاً یکی** است |
| حلقه اجرا | `src/wallex_agent/engine/loop.py` | وایرینگ، resume بعد از کرش، سیگنال‌های خاتمه |
| بک‌تست | `src/wallex_agent/backtest/runner.py` | رویدادمحور، همان هسته، متریک‌ها و منحنی equity |
| پژوهش | `docs/research.md`, `scripts/research_sweep.py` | شواهد داده‌ای پشت انتخاب استراتژی (walk-forward) |
| لاگ تصمیم | `src/wallex_agent/decision_log/logger.py` | JSONL append-only |
| CLI | `src/wallex_agent/cli.py` | `check / fetch-data / backtest / run / status / liquidate / report` |

---

## ۲. منطق استراتژی: «هسته‌ی رژیم ارزی + ماهواره‌های اختیاری»

### ۲.۱ هسته: `carry` (تخصیص USDT/TMN)
**ایده:** در بازار تومانی، دارایی با بیشترین درصد پنجره‌های ۳ روزه‌ی مثبت و کمترین نوسان، خودِ USDT است؛ سود از ضعیف شدن تومان می‌آید. ایجنت تصمیم می‌گیرد چه سهمی از سرمایه (حداکثر ۵۰٪) در USDT باشد.

روی کندل‌های **بسته‌شده‌ی ۱ ساعته‌ی USDTTMN**:

| | قاعده |
|---|---|
| رژیم روشن (کاندید خرید) | `close > EMA(72h)` **یا** بازده ۲۴ ساعته > +۰.۳٪ |
| رژیم خاموش (اشاره‌ی فروش) | `close < EMA(72h)` **و** بازده ۲۴ ساعته < −۰.۳٪ (هیسترزیس؛ جلوی نوسان بین حالت‌ها) |
| امتیاز ۰..۱ | ۰.۵×مومنتوم ۲۴h + ۰.۳×مومنتوم ۷۲h + ۰.۲×فاصله از EMA۷۲، منهای جریمه‌ی نوسان غیرعادی (جهش‌های سهمی‌شکل برمی‌گردند) |
| آستانه | امتیاز ≥ ۰.۵ تا به AI برسد |
| ریسک ماژول | سقف ۵۰٪ equity، حد ضرر دُم‌دار ۲.۵٪، بدون حد سود، بدون Time stop، افزایش تدریجی پوزیشن حداکثر هر ۶ ساعت اگر AI/رژیم اندازه‌ی بزرگ‌تری مجاز بدانند |

**خروج** (به ترتیب): تسویه پایان چالش → Kill switch → حد ضرر → رژیم خاموش. سفارش‌ها به‌صورت Maker (نشستن روی بهترین قیمت خود، کارمزد ۰.۲۵٪) و در صورت پر نشدن، Taker.

### ۲.۲ ماهواره‌ها (پیش‌فرض خاموش، با یک کلید در `settings.yaml` فعال می‌شوند)
- **`trend`**: روندگرایی ساعتی روی ارزها **به دلار** (قیمت تومانی ÷ USDTTMN تا اثر دلار جدا شود): تقاطع EMA۲۰/۱۰۰ یا شکست ۲۴ کندلی؛ حد ضرر ۳٪، حد سود ۴٪، سقف ۱۰٪. خاموش چون فقط ۳۵٪ همسایگی پارامتری‌اش خارج از نمونه مثبت ماند.
- **`mean_reversion`**: RSI و بولینگر با فیلتر روند؛ خاموش چون ۰ از ۶۴۸ ترکیب بعد از کارمزد مثبت بود.

### ۲.۳ نتایج بک‌تست با موتور واقعی (داده‌ی واقعی والکس، کارمزد Maker ۰.۲۵٪ + لغزش ۰.۰۲٪)

| اجرا | بازده خالص | بیشینه‌ی افت | معاملات | نرخ برد |
|---|---|---|---|---|
| هسته، فقط قواعد، ۲۶ سپتامبر تا ۲۲ اکتبر | **+۵.۰٪** | ۳.۹٪ | ۴ | ۷۵٪ |
| هسته، فقط قواعد، نیمه‌ی اول (تا ۱۲ اکتبر) | +۱.۰٪ | ۳.۰٪ | ۲ | ۵۰٪ |
| هسته، فقط قواعد، نیمه‌ی دوم (از ۱۲ اکتبر) | +۱.۲٪ | ۰.۲٪ | ۱ | ۱۰۰٪ |
| هسته + AI واقعی (gpt-6-sol/luna)، ۷ روز آخر | −۰.۴٪ | ۱.۱٪ | ۲ (+۳ افزایش تدریجی) | ۵۰٪ |
| بازگشت به میانگین (نسخه‌ی اولیه)، ۲۶ روز | −۲.۳٪ | ۲.۳٪ | ۹ | ۲۲٪ |

خوانش صادقانه: هسته در هر دو نیمه مثبت و کم‌افت است، اما بازده‌اش به ادامه‌ی ضعف تومان در سه روز چالش بستگی دارد (تاریخچه: ~۶۰٪ پنجره‌های ۳ روزه مثبت، دهک پایین حدود −۰.۸٪ با تخصیص ۵۰٪). اجرای ۷ روزه با AI یک استاپ در سقوط ۱۸ اکتبر خورد و بعد بازگشت را گرفت؛ همین نمونه نشان می‌دهد سیستم طبق قواعدش رفتار می‌کند، نه که سود تضمینی دارد.

## ۳. استفاده واقعی از AI (نه فقط اسم)

AI در دو نقطه‌ی تصمیم واقعی قرار دارد و خروجی‌اش **مستقیماً** روی ورود/اندازه اثر می‌گذارد، اما همیشه داخل حدود موتور ریسک:

| نقش | مدل | زمان فراخوانی | خروجی (JSON سخت) | اثر |
|---|---|---|---|---|
| **تحلیل رژیم** | `openai/gpt-6-sol` | هر ۶۰ دقیقه برای هر (ماژول، نماد) که کاندید دارد | `regime, confidence, allow_entries, size_factor, summary` | `allow_entries=false` → هیچ ورودی؛ `size_factor` سقف اندازه را کوچک می‌کند |
| **قضاوت روی کاندید** | `openai/gpt-6-luna` | برای هر کاندید با score ≥ 0.50 (شامل افزایش تدریجی) | `action (approve/approve_reduced/reject), confidence, size_factor, stop_loss_pct, take_profit_pct, rationale, key_risks` | رد → ورود نمی‌شود؛ تأیید با confidence < 0.60 → رد خودکار؛ اندازه‌ی نهایی = min(AI, رژیم) × ضریب نوسان؛ `rationale` در لاگ ذخیره می‌شود |

هر ماژول یک **بریف استراتژی** به مدل می‌دهد (`ai/analyst.py::MODULE_BRIEFS`) تا کاندید را با تز خودش بسنجد؛ مثلاً برای هسته، RSI بالا و قیمت نزدیک باند بالایی *انتظار می‌رود* و دلیل رد نیست، اما جهش سهمی‌شکل یا شکست EMA۷۲ هست. در نسخه‌ی اولیه بدون این بریف، مدل همه‌ی کاندیدهای رژیم ارزی را با الگوی «خرید در کف» رد می‌کرد؛ لاگ آن تجربه هم در تاریخچه‌ی گیت موجود است.

**قیود طراحی:**
- AI فقط می‌تواند ریسک را **کم** کند (size_factor ∈ [0,1]، استاپ فقط تنگ‌تر). موتور ریسک همه‌ی پیشنهادها را clamp می‌کند (`risk/manager.py::plan_entry`).
- AI هیچ‌وقت مستقیماً سفارش نمی‌زند؛ tool‌های MCP مربوط به سفارش/برداشت هرگز به LLM داده نمی‌شوند (`mcp_tools.py::as_openai_tools`).
- خروجی با pydantic اعتبارسنجی می‌شود؛ خروجی نامعتبر یک بار اصلاح و در غیر این صورت رد می‌شود.
- اگر LLM در دسترس نباشد، سیاست محافظه‌کار قطعی اجرا می‌شود (فقط score ≥ 0.75 با نصف سایز) و لاگ آن را با `ai_unavailable` علامت می‌زند.
- **تکرارپذیری:** همه‌ی فراخوانی‌ها (پرامپت، مدل، خروجی خام، هزینه، تأخیر) در `logs/ai_cache/<hash>.json` ذخیره می‌شوند؛ با `--ai replay` همان پاسخ‌ها بایت‌به‌بایت بازپخش می‌شوند و داور می‌تواند اجرای ما را بدون هزینه و بدون تصادفی‌بودن مدل بازسازی کند.

نمونه‌ی یک تصمیم واقعی مدل روی کاندید هسته (داده‌ی واقعی USDTTMN، ۲۱ اکتبر ۲۰۲۶، از `backtest_decisions.jsonl`):
```json
{"action":"approve_reduced","confidence":0.78,"size_factor":0.35,"stop_loss_pct":2.0,
 "rationale":"Continuation thesis is still viable: price is 0.80% above the 72h EMA and 24h return is +1.32%, with volatility not unusually high (ATR 0.24%). But the 72h return is -0.13%, and the recent 1h/6h returns are -0.04%/-0.17%; the gain reportedly came mainly from a reopening jump. Approve only at reduced size ...",
 "key_risks":["72h return remains slightly negative; momentum may be rolling over","Recent short-horizon returns are negative after a jump","Fees and 0.10% spread reduce returns; no take-profit is specified"]}
```
شش ساعت بعد، با قوی‌تر شدن رژیم، همان مدل اندازه را به ۰.۶۵ رساند و موتور ریسک یک «افزایش تدریجی» ثبت کرد (`kind: top_up`).

### ابزارهای والکس
- **Wallex REST API** (`https://api.wallex.ir`): مسیر اصلی داده و اجرا (`/v1/markets`, `/v1/depth`, `/v1/udf/history`, `/v1/account/balances`, `/v1/account/fee`, `/v1/account/orders`, `/v1/account/openOrders`).
- **Wallex MCP** (`https://api.wallex.ir/mcp`): با فلگ `--mcp`، ابزارهای فقط‌خواندنی سرور MCP والکس به تحلیل‌گر داده می‌شوند تا در تحلیل رژیم، کانتکست بیشتر (آمار بازار، اردربوک، معاملات اخیر) بگیرد. اجرای سفارش همیشه از REST و زیر نظر موتور ریسک است.

---

## ۴. مدیریت سرمایه و ریسک (`config/settings.yaml` → `risk:` و هر ماژول)

| قاعده | مقدار پیش‌فرض |
|---|---|
| سقف کل در بازار (نیمی از حساب همیشه تومان است) | ۵۰٪ equity |
| سقف هسته‌ی `carry` / ماهواره‌ها | ۵۰٪ / ۱۰٪ equity |
| حداکثر پوزیشن باز | ۲ |
| حد ضرر هسته (دُم‌دار) / ماهواره‌ی روند | ۲.۵٪ / ۳٪ |
| حد سود هسته | ندارد (خروج با رژیم) |
| توقف روزانه (بدون ورود جدید تا فردا) | Drawdown روزانه ≥ ۲٪ |
| Kill switch (بستن همه و توقف) | Drawdown کل ≥ ۴٪ |
| Cooldown بعد از استاپ / بین افزایش‌های تدریجی | ۶ ساعت / ۶ ساعت |
| حداقل اندازه سفارش | ۳۰۰٬۰۰۰ تومان |
| مقیاس نوسان | اندازه × min(1, 1/ATR%) |
| تسویه پایان چالش | ۴۵ دقیقه قبل از پایان، همه به TMN |
| اجرای سفارش | Maker روی بهترین قیمت خود (۰.۲۵٪)، بعد از ۹۰ ثانیه Taker (۰.۳٪)؛ یا MARKET |

AI فقط می‌تواند اندازه را کوچک و استاپ را تنگ کند؛ همه‌ی پیشنهادها در `risk/manager.py::plan_entry` clamp می‌شوند.

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
wallex-agent fetch-data --days 30                 # کندل‌های ساعتی واقعی والکس → data/candles/*_60.csv
wallex-agent fetch-data --days 30 --resolution 1  # کندل ۱ دقیقه‌ای (برای پژوهش؛ scripts/research_sweep.py)
wallex-agent backtest --ai off --symbols USDTTMN  # هسته، کاملاً قطعی (فقط قواعد)
wallex-agent backtest --ai off --start 2026-10-12 # walk-forward: فقط دوره‌ی آزمون
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
pytest -q        # ۱۳ تست: اندیکاتورها، سیگنال‌ها، ریسک (سایز/سقف‌ها/خروج‌ها/Drawdown)، بک‌تست end-to-end، تکرارپذیری، فالبک AI
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
| `entry` / `top_up` | **چه سیگنالی دید و چرا خرید:** ماژول، سیگنال و دلایل، رژیم، تصمیم AI و `why`، طرح ریسک (سایز/SL/TP)، equity قبل از ورود، fill واقعی |
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
