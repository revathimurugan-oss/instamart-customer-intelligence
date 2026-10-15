\# Instamart Customer Intelligence \& Support Copilot



\## Overview



This project builds an offline, synthetic-data customer intelligence and support copilot for an Instamart-style grocery delivery service.



It combines customer segmentation, lapse-risk modelling, ticket triage, policy retrieval, and a tool-using refund support agent. All experiments use local synthetic data and an offline mock language model. No live customer data, external API keys, or network-based LLM calls are required.



\*\*Key components\*\*

\- Part A: Data generation, cleaning, validation, exploratory analysis, and SQLite queries.

\- Part B: Customer segmentation and lapse-risk modelling.

\- Part C: Ticket classification, structured-output validation, retries, redaction, and token-budget experiments.

\- Part D: TF-IDF policy retrieval, policy-answer evaluation, a ReAct refund agent, tool execution, checkpointing, and safe fallbacks.



\## Reproduce



\### Requirements



\- Python 3.11 or later

\- SQLite 3.25 or later

\- Packages listed in `requirements.txt`



\### Setup on Windows PowerShell



Create a virtual environment and install dependencies:



```powershell

py -3.12 -m venv .venv312

.\\.venv312\\Scripts\\python.exe -m pip install -r requirements.txt

```



Run commands with the virtual-environment Python executable. This avoids needing to change PowerShell execution policy.



\### Run the pipeline



From the project root:



```powershell

.\\.venv312\\Scripts\\python.exe generate\_data.py

.\\.venv312\\Scripts\\python.exe clean\_data.py

.\\.venv312\\Scripts\\python.exe validate\_data.py

.\\.venv312\\Scripts\\python.exe part\_a\_eda.py

.\\.venv312\\Scripts\\python.exe part\_b.py

.\\.venv312\\Scripts\\python.exe part\_c.py

.\\.venv312\\Scripts\\python.exe part\_d.py

.\\.venv312\\Scripts\\python.exe part\_d\_agent\_run.py

```



`generate\_data.py` recreates the synthetic raw data and policy files. Run the steps in order because later parts depend on earlier outputs. The Part D agent runner imports `part\_d.py` and executes the agent scenarios separately.



\### Key outputs



\- `data/instamart.db`: local SQLite database.

\- `outputs/part\_a\_eda.txt`: Part A exploratory-analysis results.

\- `outputs/customer\_segments.csv`, `segment\_profiles.csv`, `kmeans\_metrics.csv`: segmentation outputs.

\- `outputs/retention\_list.csv`, `lapse\_model.joblib`, `rf\_top\_features.csv`: lapse-risk outputs.

\- `outputs/triage\_results.csv`, `triage\_budget\_demo.csv`, `part\_c\_run\_log.txt`: ticket-triage outputs.

\- `outputs/policy\_eval\_overlap10.csv`, `policy\_eval\_overlap0.csv`: policy retrieval evaluation.

\- `outputs/agent\_runs.jsonl`: agent run results and tool observations.

\- `outputs/refund\_ledger.jsonl`: refund and escalation tool records.

\- `state/`: JSON checkpoint files used for agent recovery.



\## Part A



Part A generates synthetic customers, products, orders, tickets, policy-evaluation questions, agent scenarios, and policy text files. It cleans and validates the data, stores relevant records in SQLite, and performs exploratory analysis.



The data-validation script has passed its checkpoint checks. The recorded clean dataset includes 1,200 customers, 40 products, and 12,309 unique order IDs.



\## Part B



Part B creates customer segments using clustering and produces segment profiles and clustering metrics. It also trains a lapse-risk model and generates a retention list and feature-importance output.



Generated model metrics and segment assignments are saved under `outputs/` for review.



\## Part C



Part C classifies support tickets with the offline mock model, validates structured responses, applies retries and fallback handling, and records redaction and token-budget experiments.



Recorded evaluation results:

\- Zero-shot accuracy: 126/147 (85.7%).

\- Few-shot accuracy: 147/147 (100%) on the evaluated examples.

\- Retry recovery: 19 responses recovered.

\- In the recorded budget demonstration, 45 tickets were not completed before the budget was exhausted.



These results describe the synthetic evaluation set and should not be interpreted as real-world production accuracy.



\## Part D



\### Policy retrieval



Policy files are split into 40-word chunks with an overlap of 10 words. TF-IDF vectors and cosine similarity rank the most relevant chunks. The answer function sends the policy context and question to the offline mock model and returns the answer and its sources.



Recorded evaluation:

\- 15 policy chunks were created.

\- With overlap 10, the expected document was retrieved in the top three for 12/12 answerable questions; expected-fact matching passed 12/12, and the two unanswerable questions triggered the fallback.

\- With overlap 0, expected-fact matching passed 10/12 answerable questions, demonstrating that chunk boundaries can affect retrieval and answer evidence.



\### Refund agent and tools



The agent uses the following tools:

\- `get\_order`: retrieves order status, amount, and delivery time from SQLite.

\- `search\_policy`: retrieves relevant policy chunks.

\- `calculate\_refund`: calculates the refund amount when delivery exceeds 45 minutes.

\- `issue\_refund`: records an automatically issued refund in the JSONL ledger.

\- `escalate\_to\_human`: records a refund request requiring human review.



Only completed orders are eligible. Refunds above INR 1,000 must be escalated rather than issued automatically. The agent uses a maximum of six model calls, a token budget, and a time limit. After each tool observation, it saves a JSON checkpoint that can be resumed.



The recorded seven-scenario run passed the expected outcome checks. The token-limit and timeout demonstrations returned their respective fallbacks. The simulated crash-recovery demonstration resumed successfully and executed `get\_order`, `search\_policy`, `calculate\_refund`, and `issue\_refund` once each across the original run and its continuation.



\### ReAct versus plan-and-execute



The implemented ReAct agent chooses a tool, observes its result, and then decides what to do next. The recorded standard scenarios used between two and five model calls for completed decisions; the deliberately looping S7 scenario reached the six-call limit and returned a fallback. Token usage varied with the length of the accumulated conversation.



Plan-and-execute could reduce repeated reasoning by creating a plan once and carrying it out, potentially lowering model calls for predictable workflows. However, it can be less adaptive when a lookup returns an unexpected result, such as a missing order or a cancelled order. For refund decisions, ReAct is a reasonable choice because the agent must inspect order status and policy evidence before deciding whether to deny, issue, or escalate. Both approaches require strict tool-level safeguards and execution limits.



\## Limitations and Next Steps



\- All data and outcomes are synthetic; performance does not establish real-world accuracy.

\- The language model is an offline mock, not a deployed production LLM.

\- The refund ledger is a local demonstration log, not a payment-system integration.

\- TF-IDF retrieval is lexical and may miss semantically relevant policy text.

\- Token counts and timing are measured in this local mock environment and may not represent production usage.

\- Before production use, add authentication, authorization, audit controls, robust database transactions, privacy reviews, human approval workflows, and integration tests.

\- Further work could compare alternative retrieval methods, expand the evaluation set, and test ReAct against plan-and-execute using the same scenarios and budgets.

