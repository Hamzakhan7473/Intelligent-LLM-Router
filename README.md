# Intelligent-LLM-Router
An intelligent routing engine that classifies user prompts (e.g., summarization, code generation, Q&amp;A) and dynamically sends them to the most suitable large language model based on cost, latency, and quality.  This project replicates and simplifies key routing capabilities used in platforms like OpenRouter and NotDiamond.ai.
# 🧠 Intelligent LLM Router

> A dynamic LLM routing engine that classifies user prompts (e.g., summarization, code generation, Q&A) and automatically sends them to the best-value language model based on cost, latency, or quality preferences.

Inspired by platforms like [OpenRouter](https://openrouter.ai) and [NotDiamond.ai](https://www.notdiamond.ai), this router helps optimize usage of multiple LLMs by intelligently selecting the most appropriate model per prompt.

---

## 🚀 Features

- 🔍 **Prompt Type Detection** – Classify user input as summarization, code generation, Q&A, etc.
- 🔁 **Dynamic Routing** – Select the best LLM model based on performance and cost.
- ⚙️ **Custom Routing Rules** – Let users prioritize **Speed**, **Cost**, or **Quality**.
- 💸 **Real-Time Metrics** – Display cost, latency, and selected model per request.
- 📊 **Evaluation Mode** – Benchmark router performance on a test prompt set.
- 📈 **Bonus**: Logs, Analytics Dashboard, and NPM Package Export.

---

## 🛠️ Tech Stack

| Layer       | Tech                        |
|-------------|-----------------------------|
| Frontend    | Next.js, TypeScript, Tailwind CSS |
| Backend     | FastAPI, Python              |
| ML Routing  | scikit-learn, regex, RandomForest |
| LLM APIs    | OpenAI, NotDiamond, Local LLMs |
| Analytics   | Custom logging or Firebase (optional) |
| Hosting     | Vercel (Frontend), Render/Fly.io (Backend) |

---

## 🧱 Project Structure

