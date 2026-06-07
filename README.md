# 🎭 RiskPaperAnalysisCrew

**Multi-agent RAG system that reads and answers questions from risk analysis papers using 5 specialized agents.**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)

---

## 📋 Overview

**RiskPaperAnalysisCrew** is a demo project showcasing the integration of **RAG (Retrieval-Augmented Generation)** with **Multi-Agent Architecture**. It loads PDF research papers about misprediction risks in deep learning, indexes them using TF-IDF, and employs 5 specialized agents working in sequence to answer questions based on the papers.

### What it demonstrates:
- ✅ **RAG Pipeline** - PDF loading, TF-IDF indexing, semantic search
- ✅ **Multi-Agent System** - 5 specialized agents in a sequential pipeline
- ✅ **LLM Integration** - DeepSeek Chat (7B) for generation
- ✅ **Conversation Memory** - Context retention across queries

---

## 🏗️ Architecture

### Complete System Architecture

<img src="complete_architecture.png" width="100%">

*Figure 1: Complete multi-agent system architecture*

### Simplified Architecture

<img src="simplified_architecture.png" width="80%">

*Figure 2: Simplified view of the agent pipeline*

### Agent Pipeline Sequence

| Step | Agent | Role |
|------|-------|------|
| 1 | 🔍 **Researcher** | Searches PDFs using TF-IDF, retrieves relevant context |
| 2 | 📊 **Analyzer** | Analyzes the question and extracted context |
| 3 | 🛡️ **Strategist** | Develops answer strategy and mitigation plans |
| 4 | ✍️ **Writer** | Writes comprehensive answer as a report |
| 5 | ✅ **Reviewer** | Validates quality and approves final answer |

### Flow Diagram

<img src="agent_flow.png" width="80%">

*Figure 3: Data flow through the system*

---

## 📦 Pre-Requisites

### 1. Downloaded Model Checkpoint
- DeepSeek Chat (7B) model
- Path: `/path/to/deepseek-chat-model/`

### 2. PDF Papers
Place your risk analysis papers in the `risk_papers/` folder:
```bash
risk_papers/
├── paper1.pdf
├── paper2.pdf
└── ...
