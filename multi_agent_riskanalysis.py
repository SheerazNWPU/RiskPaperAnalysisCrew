# multi_agent_misprediction_risk.py - Multi-Agent System for Misprediction Risk Analysis of Deep Models
import torch
import re
import warnings
import os
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
from PyPDF2 import PdfReader
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Suppress warnings
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

# ============================================================
# PDF LOADER AND DOCUMENT STORE
# ============================================================
class PDFDocumentStore:
    """Loads and stores PDF documents for retrieval"""
    
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.documents = []
        self.chunks = []
        self.vectorizer = None
        self.vectors = None
        
    def load_pdfs(self):
        """Load all PDFs from folder"""
        print(f"\n   ?? Loading PDFs from: {self.folder_path}")
        
        if not os.path.exists(self.folder_path):
            print(f"   ? Folder not found: {self.folder_path}")
            print(f"   Creating folder: {self.folder_path}")
            os.makedirs(self.folder_path)
            return False
        
        pdf_files = [f for f in os.listdir(self.folder_path) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f"   ?? No PDF files found in {self.folder_path}")
            return False
        
        print(f"   ?? Found {len(pdf_files)} PDF files")
        
        for pdf_file in pdf_files:
            filepath = os.path.join(self.folder_path, pdf_file)
            print(f"      Reading: {pdf_file}")
            
            try:
                reader = PdfReader(filepath)
                num_pages = len(reader.pages)
                file_text = ""
                
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        file_text += text + "\n"
                        self.documents.append({
                            'filename': pdf_file,
                            'page': page_num + 1,
                            'text': text
                        })
                
                # Create chunks from the full document
                chunks = self._chunk_text(file_text, pdf_file)
                self.chunks.extend(chunks)
                print(f"         ? Loaded {num_pages} pages, {len(chunks)} chunks")
                
            except Exception as e:
                print(f"         ? Error: {e}")
        
        print(f"\n   ? Total: {len(self.documents)} pages, {len(self.chunks)} chunks")
        return len(self.chunks) > 0
    
    def _chunk_text(self, text, filename, chunk_size=800):
        """Split text into overlapping chunks for better retrieval"""
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length > chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) > 100:
                    chunks.append({
                        'text': chunk_text,
                        'filename': filename,
                        'page': 1,  # Page approximation
                        'chunk_id': len(chunks)
                    })
                current_chunk = []
                current_length = 0
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) > 100:
                chunks.append({
                    'text': chunk_text,
                    'filename': filename,
                    'page': 1,
                    'chunk_id': len(chunks)
                })
        
        return chunks
    
    def build_index(self):
        """Build TF-IDF index for retrieval"""
        if not self.chunks:
            return False
        
        print(f"\n   ?? Building search index for {len(self.chunks)} chunks...")
        chunk_texts = [c['text'] for c in self.chunks]
        
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.vectors = self.vectorizer.fit_transform(chunk_texts)
        print(f"   ? Index built with {self.vectors.shape[1]} features")
        return True
    
    def search(self, query, top_k=5):
        """Search for relevant chunks using TF-IDF"""
        if self.vectorizer is None or self.vectors is None:
            return []
        
        try:
            query_vec = self.vectorizer.transform([query])
            similarities = self.vectors.dot(query_vec.T).toarray().flatten()
            top_indices = similarities.argsort()[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.05:
                    results.append({
                        'chunk': self.chunks[idx],
                        'score': float(similarities[idx]),
                        'text': self.chunks[idx]['text']
                    })
            return results
        except Exception as e:
            return []
    
    def get_context(self, query, top_k=3):
        """Get relevant context for answering questions"""
        results = self.search(query, top_k)
        if not results:
            return None
        
        context_parts = []
        for r in results:
            if r['score'] > 0.1:
                context_parts.append(
                    f"[Source: {r['chunk']['filename']}] (Relevance: {r['score']:.3f})\n{r['text'][:600]}"
                )
        
        return '\n\n---\n\n'.join(context_parts) if context_parts else None


# ============================================================
# MEMORY SYSTEM
# ============================================================
class ConversationMemory:
    def __init__(self, max_history=15):
        self.history = []
        self.max_history = max_history
    
    def add(self, role, content):
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def get_recent(self, n=5):
        return self.history[-n:] if self.history else []
    
    def get_context(self, current_query, n=3):
        recent = self.history[-n*2:]
        if not recent:
            return ""
        
        context = "Previous conversation:\n"
        for msg in recent:
            if msg["role"] == "user":
                context += f"User: {msg['content']}\n"
            else:
                context += f"Assistant: {msg['content'][:200]}\n"
        context += f"\nCurrent question: {current_query}\n"
        return context
    
    def clear(self):
        self.history = []
    
    def show(self):
        if not self.history:
            return "No conversation history."
        
        output = "\n?? CONVERSATION HISTORY:\n" + "-" * 40 + "\n"
        for msg in self.history:
            role_icon = "??" if msg["role"] == "user" else "??"
            output += f"{role_icon} [{msg['timestamp']}] {msg['role']}: {msg['content'][:100]}\n"
        return output


# ============================================================
# SPECIALIZED AGENTS FOR MISPREDICTION RISK ANALYSIS
# ============================================================
class BaseAgent:
    def __init__(self, name, llm, tokenizer, memory):
        self.name = name
        self.llm = llm
        self.tokenizer = tokenizer
        self.memory = memory
        self.system_prompt = f"You are a {name} agent specializing in misprediction risk analysis of deep learning models."
    
    def think(self, task, max_new=600, include_memory=True):
        memory_context = ""
        if include_memory:
            recent = self.memory.get_recent(3)
            if recent:
                memory_context = "\nRecent conversation:\n"
                for msg in recent:
                    memory_context += f"- {msg['role']}: {msg['content'][:100]}\n"
        
        prompt = f"{self.system_prompt}\n{memory_context}\n\nTask: {task}\n\n{self.name}:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000)
        
        if hasattr(self.llm, 'device'):
            inputs = {k: v.to(self.llm.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=max_new,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if f"{self.name}:" in response:
            response = response.split(f"{self.name}:")[-1].strip()
        
        return response


class MispredictionResearcher(BaseAgent):
    """Searches PDFs for misprediction risk information"""
    
    def __init__(self, llm, tokenizer, memory, pdf_store):
        super().__init__("MispredictionResearcher", llm, tokenizer, memory)
        self.pdf_store = pdf_store
        self.system_prompt = "You are a researcher specializing in misprediction risks of deep neural networks."
    
    def run(self, task):
        print(f"   ?? {self.name}: Searching for misprediction risk information...")
        
        # Search PDFs
        context = self.pdf_store.get_context(task, top_k=4)
        
        if context:
            print(f"   ?? Found relevant information in papers")
            return f"[Found in research papers]\n\n{context}"
        
        # Keywords for misprediction risk
        risk_keywords = {
            'uncertainty': 'Uncertainty quantification helps identify when models are likely to make mistakes.',
            'calibration': 'Model calibration ensures prediction probabilities reflect true likelihoods.',
            'out-of-distribution': 'OOD detection identifies inputs that differ from training data.',
            'adversarial': 'Adversarial examples can cause confident but wrong predictions.',
            'confidence': 'Overconfidence is a major cause of mispredictions.',
        }
        
        for key, value in risk_keywords.items():
            if key in task.lower():
                return f"[General knowledge]\n{value}"
        
        return self.think(f"Research misprediction risks for: {task}", max_new=400)


class RiskAnalyzer(BaseAgent):
    """Analyzes misprediction risks and patterns"""
    
    def __init__(self, llm, tokenizer, memory):
        super().__init__("RiskAnalyzer", llm, tokenizer, memory)
        self.system_prompt = "You are a risk analyst specializing in identifying and characterizing misprediction risks in deep learning."
    
    def run(self, information, question):
        print(f"   ?? {self.name}: Analyzing misprediction risks...")
        
        task = f"""Based on the provided information, analyze the misprediction risks.

Information:
{information}

Question: {question}

Provide analysis covering:
1. What are the key misprediction risks identified?
2. Under what conditions do these risks occur?
3. How severe are these risks (confidence levels)?
4. What factors contribute to these mispredictions?

Analysis:"""
        
        return self.think(task, max_new=600)


class MitigationStrategist(BaseAgent):
    """Suggests strategies to reduce misprediction risks"""
    
    def __init__(self, llm, tokenizer, memory):
        super().__init__("MitigationStrategist", llm, tokenizer, memory)
        self.system_prompt = "You are an expert in mitigating misprediction risks in deep learning models."
    
    def run(self, analysis, question):
        print(f"   ??? {self.name}: Developing mitigation strategies...")
        
        task = f"""Based on the risk analysis, provide mitigation strategies.

Risk Analysis:
{analysis}

Question: {question}

Provide practical mitigation strategies including:
1. Pre-deployment risk reduction methods
2. Runtime monitoring approaches
3. Fallback mechanisms
4. Recommended best practices

Mitigation Strategies:"""
        
        return self.think(task, max_new=500)


class ReportWriter(BaseAgent):
    """Writes comprehensive risk assessment reports"""
    
    def __init__(self, llm, tokenizer, memory):
        super().__init__("ReportWriter", llm, tokenizer, memory)
        self.system_prompt = "You are a technical writer specializing in risk assessment reports for AI systems."
    
    def run(self, analysis, mitigation, original_question):
        print(f"   ?? {self.name}: Writing risk assessment report...")
        
        task = f"""Write a comprehensive misprediction risk assessment report.

Original Question: {original_question}
Risk Analysis: {analysis}
Mitigation Strategies: {mitigation}

Write a clear, actionable report that includes:
1. Executive summary of risks
2. Detailed risk analysis
3. Mitigation recommendations
4. Conclusions

Report:"""
        
        return self.think(task, max_new=700)


class QualityReviewer(BaseAgent):
    """Reviews and validates the risk assessment"""
    
    def __init__(self, llm, tokenizer, memory):
        super().__init__("QualityReviewer", llm, tokenizer, memory)
        self.system_prompt = "You are a quality reviewer for AI risk assessments."
    
    def run(self, report, question):
        print(f"   ? {self.name}: Reviewing assessment quality...")
        
        task = f"""Review this misprediction risk assessment.

Question: {question}
Report: {report}

Check for:
- Technical accuracy
- Completeness of risk analysis
- Actionability of recommendations

If good, respond with "APPROVED: [report]"
If needs improvement, respond with "IMPROVED: [better report]":"""
        
        review = self.think(task, max_new=500)
        
        if "APPROVED:" in review:
            final = review.split("APPROVED:")[-1].strip()
            return final, True
        elif "IMPROVED:" in review:
            final = review.split("IMPROVED:")[-1].strip()
            return final, False
        else:
            return report, True


# ============================================================
# ORCHESTRATOR
# ============================================================
class MispredictionRiskOrchestrator:
    """Coordinates all agents for misprediction risk analysis"""
    
    def __init__(self, chat_model_path, pdf_folder):
        print("="*70)
        print("?? MISPREDICTION RISK ANALYSIS SYSTEM")
        print("   For Deep Learning Models")
        print("="*70)
        
        # Load model
        print("\n[1/6] Loading DeepSeek model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            chat_model_path, trust_remote_code=True, local_files_only=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            chat_model_path, trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
        )
        self.llm.eval()
        print("   ? Model loaded")
        
        # Create memory
        print("\n[2/6] Creating conversation memory...")
        self.memory = ConversationMemory(max_history=15)
        print("   ? Memory created")
        
        # Load PDF documents
        print("\n[3/6] Loading PDF documents...")
        self.pdf_store = PDFDocumentStore(pdf_folder)
        pdf_loaded = self.pdf_store.load_pdfs()
        
        if pdf_loaded:
            self.pdf_store.build_index()
            print("   ? PDF documents loaded and indexed")
        else:
            print("   ?? No PDF documents loaded. Using general knowledge.")
        
        # Create specialized agents
        print("\n[4/6] Creating specialized agents...")
        self.researcher = MispredictionResearcher(self.llm, self.tokenizer, self.memory, self.pdf_store)
        self.analyzer = RiskAnalyzer(self.llm, self.tokenizer, self.memory)
        self.strategist = MitigationStrategist(self.llm, self.tokenizer, self.memory)
        self.writer = ReportWriter(self.llm, self.tokenizer, self.memory)
        self.reviewer = QualityReviewer(self.llm, self.tokenizer, self.memory)
        print(f"   ? Created 5 specialized agents")
        
        # Tool agents
        print("\n[5/6] Creating tool agents...")
        self.calculator = None  # Placeholder for future tools
        print(f"   ? Tools ready")
        
        # Summary
        print("\n[6/6] System ready!")
        pdf_count = len([f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]) if os.path.exists(pdf_folder) else 0
        print(f"   ?? PDF papers: {pdf_count}")
        print(f"   ?? Text chunks: {len(self.pdf_store.chunks)}")
        
        print("\n" + "="*70)
        print("?? MISPREDICTION RISK ANALYSIS SYSTEM READY")
        print("="*70)
    
    def process(self, question):
        """Process question through all agents"""
        print(f"\n{'='*70}")
        print(f"?? Question: {question}")
        print(f"{'='*70}\n")
        
        self.memory.add("user", question)
        
        # Step 1: Research from papers
        research = self.researcher.run(question)
        
        # Step 2: Analyze risks
        analysis = self.analyzer.run(research, question)
        
        # Step 3: Develop mitigation strategies
        mitigation = self.strategist.run(analysis, question)
        
        # Step 4: Write report
        report = self.writer.run(analysis, mitigation, question)
        
        # Step 5: Review quality
        final_report, _ = self.reviewer.run(report, question)
        
        # Store in memory
        self.memory.add("assistant", final_report)
        
        print(f"\n{'='*70}")
        print(f"?? MISPREDICTION RISK ASSESSMENT:")
        print(f"{'='*70}")
        print(f"\n{final_report}\n")
        print(f"{'='*70}")
        
        return final_report
    
    def chat(self):
        """Interactive chat mode"""
        print("\n" + "="*70)
        print("?? MISPREDICTION RISK ANALYSIS CHAT")
        print("="*70)
        print("\n?? ASK ABOUT:")
        print("    Types of misprediction risks (overconfidence, calibration, OOD)")
        print("    Causes of model errors")
        print("    Risk quantification methods")
        print("    Mitigation strategies")
        print("    Specific papers you've loaded")
        print("\n?? EXAMPLE QUESTIONS:")
        print("    What are the main causes of mispredictions in deep learning?")
        print("    How can we quantify uncertainty in model predictions?")
        print("    What is model calibration and why does it matter?")
        print("    How to detect out-of-distribution inputs?")
        print("    What mitigation strategies exist for overconfidence?")
        print("\n?? COMMANDS:")
        print("    'memory' - Show conversation history")
        print("    'clear' - Clear memory")
        print("    'documents' - List loaded papers")
        print("    'quit' - Exit")
        print("="*70)
        
        while True:
            try:
                question = input("\n?? You: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n?? Goodbye!")
                    break
                
                if question.lower() == 'memory':
                    print(self.memory.show())
                    continue
                
                if question.lower() == 'clear':
                    self.memory.clear()
                    print("?? Memory cleared!")
                    continue
                
                if question.lower() == 'documents':
                    print(f"\n?? Loaded papers:")
                    seen = set()
                    for doc in self.pdf_store.documents:
                        if doc['filename'] not in seen:
                            seen.add(doc['filename'])
                            print(f"   - {doc['filename']}")
                    continue
                
                self.process(question)
                
            except KeyboardInterrupt:
                print("\n\n?? Goodbye!")
                break
            except Exception as e:
                print(f"\n? Error: {e}")
                print("   Please try again.")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    DEEPSEEK_CHAT_PATH = "/home/15t/Gul/.cache/huggingface/hub/models--deepseek-ai--deepseek-llm-7b-chat/snapshots/main/"
    PDF_FOLDER = "/home/15t/Gul/SG/cnn/risk_papers"  # Put your PDFs here
    
    print("\n" + "="*70)
    print("?? MISPREDICTION RISK ANALYSIS SYSTEM")
    print("   For Deep Learning Models")
    print("="*70)
    
    # Create folder if needed
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)
        print(f"\n?? Created folder: {PDF_FOLDER}")
        print("   Please add your misprediction risk papers here:")
        print("   - PDF files about uncertainty quantification")
        print("   - Papers on model calibration")
        print("   - Research on out-of-distribution detection")
        print("   - Studies on model confidence and overconfidence")
        print("\n   Then run the program again!")
        exit(0)
    
    agent = MispredictionRiskOrchestrator(
        chat_model_path=DEEPSEEK_CHAT_PATH,
        pdf_folder=PDF_FOLDER
    )
    agent.chat()