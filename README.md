# OpenStatica

Open-source, web-based statistical & ML platform that scales from lightweight exploratory analysis to advanced modeling — with endless extensibility.

---

## What is it?

OpenStatica is a browser-first analytics workbench. It combines classic statistics, modern machine learning, and a plugin-friendly design so teams can explore data, run tests, train models, and visualize results in one place.

---

## What it does

* Descriptive and inferential statistics (from summaries to t-tests and ANOVA)
* Ready-to-use ML workflows (supervised/unsupervised, evaluation, and selection)
* Real-time, responsive analysis with async processing
* Extensible via a modular plugin system
* Works with common data formats (CSV, Excel, JSON, Parquet, …)
* Scales from local runs to distributed and GPU-backed computation
* Optional integration with model hubs (e.g., Hugging Face)

---

## Problems it solves

* **One tool for EDA → stats → ML:** Reduces tool-hopping and context switching
* **Scales with your needs:** Start small in the browser, graduate to distributed or GPU compute
* **Extensible by design:** Add new tests, models, and visualizations as plugins
* **Fast iteration:** Async workflows keep the UI responsive during heavy jobs
* **Data compatibility:** Load from common tabular formats without friction

---

## How it works (high level)

1. **Load data in the browser** and create a session.
2. **Explore** with descriptive stats and frequency distributions.
3. **Infer** using built-in tests (e.g., t-tests, ANOVA).
4. **Model** with ML engines (train, predict, evaluate).
5. **Visualize** and interpret results; export when ready.
6. **Extend** functionality by enabling plugins.

---

## Architecture (at a glance)

* **Frontend (Web UI):**
  Lightweight, modular interface for data upload, variable selection, running analyses, and viewing results/plots.

* **API Layer:**
  Versioned REST endpoints (e.g., `/api/v1`) for data operations, statistics, ML, models, and visualization.

* **Core Runtime:**

  * **Engine Registry:** Registers and orchestrates computation engines (statistical, ML, etc.).
  * **Session Manager:** Tracks per-session data, results, and models.
  * **Plugin Manager:** Discovers, loads, and exposes plugin-provided engines and routes.

* **Computation Engines:**

  * **Statistical:** Descriptive, frequency, inferential tests (e.g., t-test, ANOVA), with effect sizes and assumptions checks.
  * **ML:** Supervised/unsupervised pipelines, evaluation utilities, optional deep learning adapters.

* **Compute Backends:**
  Local by default, with optional **distributed** and **GPU** paths for heavier jobs.

* **Services Layer:**
  Focused modules for data I/O, analysis coordination, ML orchestration, visualization prep, and export.

* **Model Hub Integration (optional):**
  Hooks to pull, cache, and serve models from external repositories.

---

## Who is it for?

* Data analysts and scientists needing an end-to-end, web-based toolkit
* Educators and teams teaching or standardizing statistical workflows
* Developers who want to extend a stats/ML platform via plugins

---

## Status & Roadmap (short)

* **Core stats:** Complete
* **ML workflows:** In progress (training, evaluation, selection, AutoML plans)
* **Advanced features planned:** Distributed compute, real-time collaboration, cloud deployment, model hub integration
* **Enterprise (future):** Auth/SSO, workspaces, audit logs, enterprise-grade plugins

---

## License

MIT — free to use, modify, and extend.
