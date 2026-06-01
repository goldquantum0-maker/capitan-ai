# app.py – CAPITAN AI · ELITE INTELLIGENCE CORE v4.2 · Streaming Fix
# Sovereign AI Technologies · Osinachi Chukwu
# ═══════════════════════════════════════════════════════════════

import os, re, json, uuid, time, subprocess, tempfile, requests, streamlit as st
import xml.etree.ElementTree as ET
import numpy as np
from typing import List, Dict, Any, Optional, Generator, Tuple, Union
from datetime import datetime, timedelta
from collections import defaultdict
import math, base64
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError

from dotenv import load_dotenv
load_dotenv()

# ── Optional imports ──────────────────────────────────────────
FAISS_AVAILABLE = False
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    pass

OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    pass

LOCAL_EMBEDDING_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    LOCAL_EMBEDDING_AVAILABLE = True
except ImportError:
    pass

PLOTLY_AVAILABLE = False
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    pass

# ═══════════════════════════════════════════════════════════════
# BRANDING — Nautical Anchor Logo
# ═══════════════════════════════════════════════════════════════
APP_NAME    = "CAPITAN AI"
APP_TAGLINE = "Global Finance · Quant · Quantum · Coding · Markets · Africa"

CAPITAN_LOGO_SVG = """<svg width="36" height="36" viewBox="0 0 36 36" xmlns="http://www.w3.org/2000/svg">
  <circle cx="18" cy="7" r="3" fill="none" stroke="#4ade80" stroke-width="1.9"/>
  <line x1="18" y1="10" x2="18" y2="28.5" stroke="#4ade80" stroke-width="2.3" stroke-linecap="round"/>
  <line x1="10" y1="13.5" x2="26" y2="13.5" stroke="#4ade80" stroke-width="2" stroke-linecap="round"/>
  <path d="M18,28.5 C15,28.5 9,27 8,23" fill="none" stroke="#4ade80" stroke-width="2.1" stroke-linecap="round"/>
  <line x1="5" y1="23" x2="11" y2="23" stroke="#4ade80" stroke-width="2" stroke-linecap="round"/>
  <path d="M18,28.5 C21,28.5 27,27 28,23" fill="none" stroke="#4ade80" stroke-width="2.1" stroke-linecap="round"/>
  <line x1="25" y1="23" x2="31" y2="23" stroke="#4ade80" stroke-width="2" stroke-linecap="round"/>
</svg>"""
CAPITAN_LOGO_BASE64 = "data:image/svg+xml;base64," + base64.b64encode(CAPITAN_LOGO_SVG.encode()).decode()

# ═══════════════════════════════════════════════════════════════
# PERSISTENT STATE
# ═══════════════════════════════════════════════════════════════
STATE_FILE       = "capitan_state.json"
PROJECTS_FILE    = "capitan_projects.json"
GOALS_FILE       = "capitan_goals.json"
MEMORY_META_PATH = "capitan_memory_meta.json"
MEMORY_INDEX_PATH= "capitan_faiss.index"

def load_persistent_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE,'r') as f:
                d = json.load(f)
                return (
                    d.get('messages',[]),
                    d.get('is_pro',False),
                    d.get('is_founder',False),
                    d.get('chat_history',[]),
                    d.get('verified_txids',[]),
                    d.get('user_preferences',{}),
                    d.get('daily_count',0),
                    d.get('daily_reset',datetime.now().isoformat())
                )
        except:
            pass
    return [], False, False, [], [], {}, 0, datetime.now().isoformat()


def save_persistent_state(messages, is_pro, is_founder, chat_history=None,
                          verified_txids=None, user_preferences=None,
                          daily_count=0, daily_reset=None):
    try:
        with open(STATE_FILE,'w') as f:
            json.dump({
                'messages': messages,
                'is_pro': is_pro,
                'is_founder': is_founder,
                'chat_history': chat_history or [],
                'verified_txids': verified_txids or [],
                'user_preferences': user_preferences or {},
                'daily_count': daily_count,
                'daily_reset': daily_reset or datetime.now().isoformat()
            }, f, indent=2)
    except:
        pass


def persist_current_state():
    save_persistent_state(
        st.session_state.messages,
        st.session_state.is_pro,
        st.session_state.is_founder,
        st.session_state.chat_history,
        st.session_state.verified_txids,
        st.session_state.user_preferences,
        st.session_state.daily_count,
        st.session_state.daily_reset
    )
    save_projects(st.session_state.projects)
    save_goals(st.session_state.goals)


def load_projects():
    if os.path.exists(PROJECTS_FILE):
        try:
            with open(PROJECTS_FILE,'r') as f: return json.load(f)
        except: pass
    return {}

def save_projects(p):
    try:
        with open(PROJECTS_FILE,'w') as f: json.dump(p,f,indent=2)
    except: pass

def load_goals():
    if os.path.exists(GOALS_FILE):
        try:
            with open(GOALS_FILE,'r') as f: return json.load(f)
        except: pass
    return []

def save_goals(g):
    try:
        with open(GOALS_FILE,'w') as f: json.dump(g,f,indent=2)
    except: pass

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
CONFIG = {
    "OPENROUTER_KEY":    os.environ.get("OPENROUTER_API_KEY",""),
    "OPENAI_API_KEY":    os.environ.get("OPENAI_API_KEY",""),
    "SERPER_KEY":        os.environ.get("SERPER_API_KEY",""),
    "WOLFRAM_APP_ID":    os.environ.get("WOLFRAM_APP_ID",""),
    "ETHERSCAN_API_KEY": os.environ.get("ETHERSCAN_API_KEY",""),
    "HF_TOKEN":          os.environ.get("HF_TOKEN",""),

    "PRO_MODELS":     ["anthropic/claude-3.5-sonnet","openai/gpt-4o",
                       "deepseek/deepseek-r1","google/gemini-pro-1.5"],
    "FREE_MODELS":    ["deepseek/deepseek-chat","meta-llama/llama-3.1-70b-instruct"],
    "FAST_MODEL":     "deepseek/deepseek-chat",
    "SMART_MODEL":    "anthropic/claude-3.5-sonnet",
    "DEEP_MODEL":     "deepseek/deepseek-r1",
    "PLANNER_MODEL":  "deepseek/deepseek-r1",
    "CRITIC_MODEL":   "anthropic/claude-3.5-sonnet",
    "REFINER_MODEL":  "anthropic/claude-3.5-sonnet",
    "SYNTH_MODEL":    "anthropic/claude-3.5-sonnet",
    "COMPUTE_MODEL":  "deepseek/deepseek-r1",

    "CREATOR":        "Sovereign AI Technologies · Osinachi Chukwu",
    "FOUNDER_KEY":    os.environ.get("FOUNDER_KEY","cap-founder-key"),

    "CIRCUIT_BREAKER_TIMEOUTS": {
        "web_search":3,"live_prices":5,"llm_call":60,"news":8,"crypto_verify":10,
    },
    "PRICE_CACHE_TTL":         30,
    "NEWS_CACHE_TTL":          300,
    "MEMORY_DECAY_HALF_LIFE":  30,
    "SIMPLE_WORD_THRESHOLD":   12,
    "COMPLEX_WORD_THRESHOLD":  35,
    "DEEP_THINK_DOMAINS":      {"quant","quantum","finance","macro","coding","african_finance"},

    "MULTI_AGENT_THRESHOLD":   "standard",
    "ADVERSARIAL_THRESHOLD":   "deep",
    "CONFIDENCE_DECIMALS":     1,
    "MAX_REFINEMENT_ROUNDS":   2,
    "HYPOTHESIS_COUNT":        3,

    "FREE_DAILY_LIMIT": 100,
}

CRYPTO_ADDRESSES = {
    "BTC":  "bc1qrv6yr6e0mat96rvrc8smdf9rvu9rlp8xuk8new",
    "ETH":  "0x5bd39ad3e8b1cb01e7385958160fd9b2675d02d1",
    "USDC": "0x5bd39ad3e8b1cb01e7385958160fd9b2675d02d1",
    "SOL":  "59RMErx3YYKoqdSKeMoNG5FQUX1BNyw2Rh4VPdFfrTT1",
}
PRO_PRICE_CRYPTO = {"BTC":0.00025,"ETH":0.005,"USDC":15,"SOL":0.1}
PRO_PRICE_USD    = 15
EXPLORER_LINKS   = {
    "BTC":"https://www.blockchain.com/explorer/transactions/btc/",
    "ETH":"https://etherscan.io/tx/","USDC":"https://etherscan.io/tx/",
    "SOL":"https://solscan.io/tx/",
}

# ═══════════════════════════════════════════════════════════════
# CIRCUIT BREAKER
# ═══════════════════════════════════════════════════════════════
class CircuitBreaker:
    def __init__(self, name, timeout=5, max_failures=3, reset_timeout=60):
        self.name=name; self.timeout=timeout; self.max_failures=max_failures
        self.reset_timeout=reset_timeout; self.failures=0
        self.last_failure=None; self.state="closed"

    def call(self, func, *args, **kwargs):
        if self.state=="open":
            if self.last_failure and datetime.now()-self.last_failure > timedelta(seconds=self.reset_timeout):
                self.state="half-open"
            else:
                return None, f"{self.name} temporarily unavailable"
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(func, *args, **kwargs)
                result = future.result(timeout=self.timeout)
            if self.state=="half-open": self.state="closed"; self.failures=0
            return result, None
        except FutureTimeoutError:
            self.failures+=1; self.last_failure=datetime.now()
            if self.failures>=self.max_failures: self.state="open"
            return None, f"{self.name} timed out after {self.timeout}s"
        except Exception as e:
            self.failures+=1; self.last_failure=datetime.now()
            if self.failures>=self.max_failures: self.state="open"
            return None, f"{self.name} failed: {str(e)[:100]}"

web_search_cb  = CircuitBreaker("web_search",  timeout=3)
live_prices_cb = CircuitBreaker("live_prices", timeout=5)
news_cb        = CircuitBreaker("news",        timeout=8)
llm_cb         = CircuitBreaker("llm",         timeout=60, max_failures=3)

# ═══════════════════════════════════════════════════════════════
# ENTITY MEMORY
# ═══════════════════════════════════════════════════════════════
class EntityMemory:
    def __init__(self): self.entities = {}
    def extract_entities(self, text):
        entities = []
        for p in [r'(?:my|our|the)\s+(?:company|startup|business|firm)\s+(?:is\s+)?(?:called\s+)?["\']?([A-Z][A-Za-z0-9\s&]+(?:Inc|Ltd|LLC|Capital|Ventures|Technologies)?)["\']?',
                  r'(?:at|for|with)\s+([A-Z][A-Za-z0-9]+(?:\s(?:Inc|Ltd|LLC|Capital|Technologies|Bank|Group|Holdings)))']:
            for m in re.findall(p,text): entities.append({"type":"company","name":m.strip()})
        for loc in ["Lagos","Accra","Nairobi","Johannesburg","Cairo","Abuja","London","New York","Dubai","Singapore","Ghana","Nigeria","Kenya","South Africa","Egypt"]:
            if loc.lower() in text.lower(): entities.append({"type":"location","name":loc})
        for ind in ["fintech","agritech","healthtech","edtech","logistics","payments","banking","insurance","investment","real estate","agriculture","energy","telecom","media"]:
            if ind.lower() in text.lower(): entities.append({"type":"industry","name":ind})
        return entities
    def add_entities(self, entities):
        for e in entities:
            k=f"{e['type']}:{e['name'].lower()}"
            if k not in self.entities: self.entities[k]={"type":e["type"],"name":e["name"],"first_seen":datetime.now().isoformat(),"mention_count":1}
            else: self.entities[k]["mention_count"]+=1; self.entities[k]["last_seen"]=datetime.now().isoformat()
    def get_summary(self):
        if not self.entities: return ""
        bt=defaultdict(list)
        for k,e in self.entities.items(): bt[e["type"]].append(e)
        lines=["USER ENTITIES:"]
        for et,es in bt.items():
            top=sorted(es,key=lambda x:x["mention_count"],reverse=True)[:3]
            lines.append(f"  {et.capitalize()}: {', '.join(e['name'] for e in top)}")
        return "\n".join(lines)

entity_memory = EntityMemory()

# ═══════════════════════════════════════════════════════════════
# GOAL TRACKER
# ═══════════════════════════════════════════════════════════════
class GoalTracker:
    def __init__(self): self.goals=load_goals()
    def add_goal(self,d,c="general"):
        g={"id":str(uuid.uuid4()),"description":d,"category":c,"status":"active","created":datetime.now().isoformat(),"progress":[]}
        self.goals.append(g); save_goals(self.goals); return g
    def get_active_goals(self): return [g for g in self.goals if g["status"]=="active"]
    def get_context_for_ai(self):
        a=self.get_active_goals()
        if not a: return ""
        lines=["USER GOALS:"]
        for g in a[:5]:
            da=(datetime.now()-datetime.fromisoformat(g["created"])).days
            lines.append(f"  • {g['description']} ({da}d ago)")
        return "\n".join(lines)

goal_tracker = GoalTracker()

# ═══════════════════════════════════════════════════════════════
# ELITE DOMAIN ROUTER
# ═══════════════════════════════════════════════════════════════
class DomainRouter:
    TRADING = [r'\b(swing trade|day trade|scalp|entry price|stop.?loss|take.?profit|tp|sl)\b',r'\b(when (?:to|should i) (?:buy|sell)|is now a good time|buy signal|sell signal)\b']
    CODING = [r'```',r'\bdef\s+\w+\s*\(',r'class\s+\w+.*:',r'\b(write|implement|build|create|code|refactor|debug|optimize|review)\b.*\b(function|class|api|algorithm|script|program|service|module|library)\b',r'\b(python|numpy|pandas|rust|go|golang|java|typescript|javascript|sql|c\+\+|c#|swift|kotlin)\b',r'\b(big.?o|time complexity|space complexity|algorithm|data structure|design pattern|architecture|microservice|api|rest|graphql|grpc)\b']
    QUANT = [r'\b(monte carlo|black.scholes|ito.?lemma|stochastic|option pricing|greeks|delta|gamma|theta|vega|rho)\b',r'\b(var|cvar|expected shortfall|risk management|markowitz|sharpe|sortino|calmar|information ratio)\b',r'\b(backtest|factor model|alpha|beta|capm|fama.french|momentum|mean reversion|pairs trading|stat arb)\b']
    QUANTUM = [r'\b(quantum|qubit|qiskit|qaoa|vqe|entanglement|superposition|quantum circuit|quantum gate|bell state|bloch sphere)\b']
    AFRICAN = [r'\b(african|africa|nigeria|ghana|kenya|south africa|ethiopia|egypt|morocco|tanzania|uganda|angola|ivory coast|senegal)\b',r'\b(naira|rand|cedi|shilling|pound|dirham|franc|birr|kwanza|ngn|zar|kes|egp|mad|xof)\b',r'\b(ngx|jse|gse|nse|brvm|egx|masi|afcfta|ecowas|sadc|au|mtn|dangote|ecobank|zenith|gtco|equity bank|safaricom)\b']
    MACRO = [r'\b(gdp|gnp|recession|inflation|deflation|stagflation|fiscal policy|monetary policy|central bank|fed|ecb|boj|pboc|boe)\b',r'\b(interest rate|yield curve|quantitative easing|tightening|balance sheet|money supply|m2|credit cycle|business cycle)\b']
    FINANCE = [r'\b(revenue|earnings|ebitda|ebit|fcf|free cash flow|valuation|pe ratio|pb ratio|ev.?ebitda|dcf|wacc|irr|npv)\b',r'\b(stock|bond|equity|debt|credit|yield|spread|ipo|m&a|acquisition|merger|private equity|venture capital)\b',r'\b(bitcoin|ethereum|crypto|btc|eth|defi|nft|blockchain|web3|dao|staking|yield farming)\b']
    MATH = [r'\b(prove|proof|theorem|lemma|corollary|derive|derivation|integral|derivative|gradient|hessian|jacobian)\b',r'\b(differential equation|ode|pde|fourier|laplace|linear algebra|eigenvalue|eigenvector|svd|matrix decomposition)\b']
    SOCIAL = [r'\b(hello|hi|hey|how are you|good morning|good afternoon|good evening|happy sunday|happy monday|happy tuesday|happy wednesday|happy thursday|happy friday|happy saturday)\b',r'\b(how.s it going|what.s up|how do you do|nice to meet you)\b',r'\b(thank you|thanks|appreciate|grateful|you.re amazing|great job)\b',r'\b(i.m feeling|i feel|i am feeling|feeling kinda|feeling a bit|been feeling)\b',r'\b(tired|sad|lonely|stressed|anxious|worried|overwhelmed|happy|excited)\b']
    CAPABILITIES = [r'\b(what can you do|capabilities|features|what are you|tell me about yourself|who are you|what do you do|how do you work|how can you help)\b']

    @classmethod
    def classify(cls, q):
        q_low = q.lower()
        for p in cls.SOCIAL:
            if re.search(p, q_low): return "general"
        for p in cls.CAPABILITIES:
            if re.search(p, q_low): return "general"
        for p in cls.TRADING:
            if re.search(p, q_low): return "trading_refuse"
        for p in cls.MATH:
            if re.search(p, q_low): return "math"
        for p in cls.CODING:
            if re.search(p, q_low): return "coding"
        for p in cls.QUANT:
            if re.search(p, q_low): return "quant"
        for p in cls.QUANTUM:
            if re.search(p, q_low): return "quantum"
        for p in cls.AFRICAN:
            if re.search(p, q_low): return "african_finance"
        for p in cls.MACRO:
            if re.search(p, q_low): return "macro"
        for p in cls.FINANCE:
            if re.search(p, q_low): return "finance"
        return "general"

# ═══════════════════════════════════════════════════════════════
# QUERY COMPLEXITY ANALYZER
# ═══════════════════════════════════════════════════════════════
class QueryComplexityAnalyzer:
    DEEP_PATTERNS = [r'\b(derive|prove|demonstrate|rigorously|formally|mathematically)\b',r'\b(mechanism|causal(ity|ly)?|why exactly|root cause|underlying)\b',r'\b(optimize|maximiz|minimiz|equilibrium|optimal|pareto)\b',r'\b(model|simulation|backtest|regression|forecast|predict)\b',r'\b(comprehensive|exhaustive|thorough|in.depth|detailed|full analysis)\b',r'\b(compare|contrast|versus|vs\.?|trade.?off|pros and cons)\b',r'\b(design|architect|system|framework|infrastructure|pipeline)\b',r'\?.*\?']
    SIMPLE_PATTERNS = [r'^(what is|what are|define|who is|when was|where is)\b',r'^(how much|how many|what does|does)\b']

    @classmethod
    def grade(cls, query, domain):
        words = len(query.split())
        q_low = query.lower()
        for p in cls.SIMPLE_PATTERNS:
            if re.search(p, q_low) and words < 15: return "simple"
        if words < CONFIG["SIMPLE_WORD_THRESHOLD"]: return "simple"
        deep_score = 0
        for p in cls.DEEP_PATTERNS:
            if re.search(p, q_low): deep_score += 1
        if domain in CONFIG["DEEP_THINK_DOMAINS"]: deep_score += 1
        if words > CONFIG["COMPLEX_WORD_THRESHOLD"]: deep_score += 1
        if "step by step" in q_low or "walk me through" in q_low: deep_score += 2
        if deep_score >= 3: return "deep"
        if deep_score >= 1 or words > 20: return "standard"
        return "simple"

# ═══════════════════════════════════════════════════════════════
# ELITE REASONING ENGINE
# ═══════════════════════════════════════════════════════════════
class EliteReasoningEngine:
    SOCRATIC_PROMPT = """You are the world's most rigorous reasoning architect.
QUERY: {query} | DOMAIN: {domain} | COMPLEXITY: {complexity}
Output ONLY valid JSON with: core_epistemic_question, hidden_assumptions, atomic_sub_problems, competing_hypotheses, critical_distinctions, base_rate_anchors, second_order_effects, potential_reasoning_failures, confidence_limiting_factors, elite_answer_structure, domain_specific_frameworks."""

    ADVERSARIAL_CRITIC_PROMPT = """Brutally critique this answer. QUESTION: {question} | ANSWER: {answer}
Output ONLY valid JSON with: verdict (WEAK|ACCEPTABLE|STRONG), overall_score (1-10), critical_flaws, missing_insights, confidence_issues, strongest_counterargument, what_an_expert_would_add, one_sentence_improvement."""

    ELITE_SYNTHESIS_PROMPT = """Synthesize FINAL ELITE ANSWER. QUESTION: {question} | SCAFFOLD: {scaffold} | CRITIQUE: {critique} | INITIAL: {initial_answer}
Fix every HIGH-severity flaw. Add missing insights. Respond to strongest counterargument. Do not mention this is refined."""

    @classmethod
    def decompose(cls, query, domain, complexity, is_pro):
        try:
            prompt = cls.SOCRATIC_PROMPT.format(query=query, domain=domain, complexity=complexity)
            r, err = llm_cb.call(call_llm, [{"role":"system","content":"Output only valid JSON."},{"role":"user","content":prompt}], is_pro=is_pro, use_specific_model=CONFIG["PLANNER_MODEL"])
            if err: return {}
            m = re.search(r'\{.*\}', r, re.DOTALL)
            if m: return json.loads(m.group())
        except: pass
        return {}

    @classmethod
    def critique(cls, question, answer, is_pro):
        try:
            prompt = cls.ADVERSARIAL_CRITIC_PROMPT.format(question=question, answer=answer[:3000])
            r, err = llm_cb.call(call_llm, [{"role":"system","content":"Output only valid JSON."},{"role":"user","content":prompt}], is_pro=is_pro, use_specific_model=CONFIG["CRITIC_MODEL"])
            if err: return {}
            m = re.search(r'\{.*\}', r, re.DOTALL)
            if m: return json.loads(m.group())
        except: pass
        return {}

    @classmethod
    def synthesize_elite(cls, question, scaffold_text, critique_data, initial_answer, is_pro):
        try:
            critique_text = json.dumps(critique_data, indent=2) if critique_data else "None"
            prompt = cls.ELITE_SYNTHESIS_PROMPT.format(question=question, scaffold=scaffold_text[:2000], critique=critique_text[:1500], initial_answer=initial_answer[:3000])
            r, err = llm_cb.call(call_llm, [{"role":"system","content":"You are an elite expert."},{"role":"user","content":prompt}], is_pro=is_pro, use_specific_model=CONFIG["REFINER_MODEL"])
            return r if not err else initial_answer
        except: return initial_answer

    @classmethod
    def build_scaffold_context(cls, plan):
        if not plan: return ""
        lines = ["=== ELITE REASONING SCAFFOLD ==="]
        if plan.get("core_epistemic_question"): lines.append(f"CORE: {plan['core_epistemic_question']}")
        if plan.get("hidden_assumptions"): lines.append("HIDDEN ASSUMPTIONS: " + "; ".join(plan["hidden_assumptions"]))
        if plan.get("competing_hypotheses"):
            lines.append("COMPETING HYPOTHESES:")
            for h in plan["competing_hypotheses"]: lines.append(f"  H: {h.get('hypothesis','')} [P≈{h.get('prior_probability','?')}]")
        if plan.get("critical_distinctions"): lines.append("CRITICAL DISTINCTIONS: " + "; ".join(plan["critical_distinctions"]))
        if plan.get("base_rate_anchors"): lines.append("BASE RATES: " + "; ".join(plan["base_rate_anchors"]))
        lines.append("=== END SCAFFOLD ===")
        return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════
# ELITE SELF-EVALUATOR
# ═══════════════════════════════════════════════════════════════
class EliteSelfEvaluator:
    @staticmethod
    def evaluate(q, a, domain="general", is_pro=False):
        try:
            prompt = f"""Peer reviewer. QUESTION: {q} | ANSWER: {a[:3000]} | DOMAIN: {domain}
Score 1.0-5.0. Return ONLY valid JSON with: accuracy, completeness, logical_rigor, evidence_quality, calibration, intellectual_honesty, practical_utility, domain_depth, second_order_thinking, communication_clarity, novel_insight, adversarial_robustness, confidence, weakest_dimension, key_weakness, missing_insight, highest_impact_improvement, expert_would_say, brief_justification."""
            r, err = llm_cb.call(call_llm, [{"role":"system","content":"Output only valid JSON."},{"role":"user","content":prompt}], is_pro=is_pro, use_specific_model=CONFIG["CRITIC_MODEL"])
            if err: return EliteSelfEvaluator._defaults()
            m = re.search(r'\{.*\}', r, re.DOTALL)
            if m: return json.loads(m.group())
        except: pass
        return EliteSelfEvaluator._defaults()

    @staticmethod
    def _defaults():
        return {"accuracy":3.0,"completeness":3.0,"logical_rigor":3.0,"evidence_quality":2.5,"calibration":3.0,"intellectual_honesty":3.0,"practical_utility":3.0,"domain_depth":3.0,"second_order_thinking":2.5,"communication_clarity":3.5,"novel_insight":2.5,"adversarial_robustness":2.5,"confidence":3.0,"weakest_dimension":"evidence_quality","key_weakness":"N/A","missing_insight":"N/A","highest_impact_improvement":"N/A","expert_would_say":"N/A","brief_justification":"Baseline evaluation."}

# ═══════════════════════════════════════════════════════════════
# COMPUTATIONAL ENGINE
# ═══════════════════════════════════════════════════════════════
class ComputationalEngine:
    @staticmethod
    def query_wolfram(expression):
        if not CONFIG.get("WOLFRAM_APP_ID"): return None
        try:
            r = requests.get("http://api.wolframalpha.com/v2/query", params={"input":expression,"appid":CONFIG["WOLFRAM_APP_ID"],"output":"json","format":"plaintext"}, timeout=10)
            if r.status_code != 200: return None
            pods = r.json().get("queryresult",{}).get("pods",[])
            results = []
            for pod in pods[:4]:
                for sub in pod.get("subpods",[]):
                    text = sub.get("plaintext","").strip()
                    if text and len(text) > 1: results.append(f"{pod.get('title','')}: {text}")
            return "\n".join(results) if results else None
        except: return None

    @staticmethod
    def execute_code(code, timeout=20):
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f: f.write(code); tmp = f.name
            try:
                p = subprocess.run(["python3", tmp], capture_output=True, text=True, timeout=timeout, env={**os.environ,"PYTHONPATH":os.getcwd()})
                return {"stdout":p.stdout,"stderr":p.stderr,"returncode":p.returncode,"success":p.returncode==0}
            except subprocess.TimeoutExpired: return {"success":False,"stderr":"Timeout (20s)","stdout":"","returncode":-1}
            finally: os.unlink(tmp)
        except Exception as e: return {"success":False,"stderr":str(e),"stdout":"","returncode":-1}

# ═══════════════════════════════════════════════════════════════
# REFINED PERSONAS
# ═══════════════════════════════════════════════════════════════

ELITE_CORE = """
╔══════════════════════════════════════════════════════════════╗
║          CAPITAN AI · ELITE INTELLIGENCE CORE v4.2          ║
║          Sovereign AI Technologies · Osinachi Chukwu        ║
╚══════════════════════════════════════════════════════════════╝

ELITE REASONING PRINCIPLES:
1. MECHANISM FIRST — State what AND explain why/how with causal chain.
2. CALIBRATED CONFIDENCE — Assign explicit confidence: "~70% confident based on [source]"
3. EVIDENCE TRACEABILITY — Cite sources: NBS, CBN, World Bank, Yahoo Finance, or reasoning chain.
4. STEEL-MAN OPPONENTS — Present strongest counterargument, then respond.
5. QUANTIFY — Numbers, ranges, magnitudes. Not "significant" but "47% increase"
6. SECOND-ORDER THINKING — Effects at least 2 levels deep.
7. AFRICAN MARKET DEPTH — Apply Africa-specific dynamics where relevant.
8. INTELLECTUAL HONESTY — Flag what you don't know. Say "this is speculative" when it is.

COMMUNICATION:
• MATURE — Direct. No performative enthusiasm. No "Great question!" No exclamation marks.
• AUTHENTIC — Trusted colleague, not customer service. Earn trust through accuracy.
• SIMPLE — Short sentences. Plain language. One idea per paragraph.
• NO FORCED WARMTH — Do NOT use: "my friend," "Ah," "I see you," "oya," West African colloquialisms.
• GREETINGS — "Hello. How can I help?" — then get to the substance.
"""

REFINED_GENERAL = """You are CAPITAN AI — direct, knowledgeable, genuinely helpful.
CORE IDENTITY: A trusted colleague who knows their field deeply and shares knowledge clearly.
GREETINGS: "Hello. How can I help?" — then get to the substance.
EMOTIONAL: Acknowledge briefly: "That sounds difficult." Then offer practical support.
YOUR VOICE: Direct. Clear. Calm. Honest. Helpful.
NEVER: "my friend," "ah," "I see you," "oya," "Great question!," "Certainly!," "Absolutely!"
"""

PERSONAS = {
    "trading_refuse": ELITE_CORE + """DOMAIN: TRADING SIGNALS — REFUSAL POLICY.
You do NOT provide specific entry prices, stop-losses, or take-profit levels.
Explain clearly why: frameworks empower traders to think independently; signals create dependency and liability.
Redirect to structural analysis — macro regime, valuation, risk frameworks, educational content.
Be direct and respectful. Do not preach.""",
    "coding": ELITE_CORE + "DOMAIN: SOFTWARE ENGINEERING. Role: Principal Engineer. Production-quality code with type hints, docstrings, tests, complexity analysis. Propose design before code. Flag security issues. Be direct and helpful.",
    "quant": ELITE_CORE + "DOMAIN: QUANTITATIVE FINANCE. Role: Quant Research Director. Assumption audit, mathematical derivation, vectorised implementation, validation. NEVER provide entry/exit signals.",
    "quantum": ELITE_CORE + "DOMAIN: QUANTUM COMPUTING. Role: Quantum Principal Scientist. Full Dirac notation, circuit diagrams, NISQ-era realism, Qiskit/Cirq code.",
    "finance": ELITE_CORE + "DOMAIN: GLOBAL FINANCE & INVESTMENT. Role: Goldman Sachs MD + Bridgewater Analyst. Macro regime, dual valuation, probability-weighted scenarios. Use live prices. Cite data sources. NEVER provide entry/exit levels.",
    "african_finance": ELITE_CORE + "DOMAIN: AFRICAN FINANCIAL MARKETS. Role: Africa's Premier Finance Intelligence — NGX, JSE, GSE, BRVM, NSE, EGX, MASI. Sovereign macro, FX risk architecture, Africa-adjusted valuations. Cite NBS, CBN, World Bank data. NEVER provide entry/exit levels.",
    "macro": ELITE_CORE + "DOMAIN: GLOBAL MACRO ECONOMICS. Role: Global Macro PM. Regime identification, CB reaction function, fiscal sustainability, cross-asset. Cite data sources. NEVER provide entry/exit levels.",
    "math": ELITE_CORE + "DOMAIN: PURE & APPLIED MATHEMATICS. Role: Research Mathematician. Full derivations, rigorous proofs, SymPy verification, edge case analysis.",
    "general": ELITE_CORE + REFINED_GENERAL,
}

# ═══════════════════════════════════════════════════════════════
# LLM CALLERS — FIXED: No duplicate words in streaming
# ═══════════════════════════════════════════════════════════════

def _get_api_key():
    key = CONFIG.get("OPENROUTER_KEY", "").strip()
    if key and key.startswith("sk-or-"): return key
    for name in ["OPENROUTER_API_KEY", "openrouter_api_key", "OPENROUTER_KEY"]:
        val = os.environ.get(name, "").strip()
        if val and val.startswith("sk-or-"): return val
    try:
        if hasattr(st, "secrets"):
            for name in ["OPENROUTER_API_KEY", "openrouter_api_key", "OPENROUTER_KEY"]:
                if name in st.secrets:
                    val = str(st.secrets[name]).strip()
                    if val and val.startswith("sk-or-"): return val
    except: pass
    return ""


def call_llm(messages, is_pro=False, use_specific_model=None):
    api_key = _get_api_key()
    if not api_key:
        return "**Configuration required:** OpenRouter API key not found.\n\n1. Go to openrouter.ai/keys\n2. Add OPENROUTER_API_KEY to Streamlit Secrets\n3. Reboot"
    if use_specific_model: models = [use_specific_model]
    elif is_pro: models = CONFIG["PRO_MODELS"] + CONFIG["FREE_MODELS"]
    else: models = CONFIG["FREE_MODELS"]
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for model in models:
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json={"model":model,"messages":messages,"temperature":0.2,"max_tokens":1024}, timeout=30)
            if r.status_code == 200: return r.json()["choices"][0]["message"]["content"]
            if r.status_code == 401: return "API key invalid. Get new one at openrouter.ai/keys"
        except: continue
    return "Unable to reach any model. Check your connection and API key."


def call_llm_stream_fast(messages, is_pro=False, model_override=None):
    """Stream response — FIXED: yields complete accumulated text, no duplicates."""
    api_key = _get_api_key()
    if not api_key:
        yield "**Configuration required:** OpenRouter API key not found. Add OPENROUTER_API_KEY to Streamlit Secrets."
        return

    if model_override: models = [model_override]
    elif is_pro: models = CONFIG["PRO_MODELS"] + CONFIG["FREE_MODELS"]
    else: models = CONFIG["FREE_MODELS"]

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for model in models:
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json={"model": model, "messages": messages, "temperature": 0.2, "max_tokens": 1024, "stream": True},
                timeout=180,
                stream=True,
            )
            if r.status_code != 200:
                continue

            full_text = ""
            for line in r.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        d = line[6:]
                        if d == "[DONE]":
                            break
                        try:
                            delta = json.loads(d).get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if delta:
                                full_text += delta
                        except:
                            continue

            # Yield the complete text ONCE — no chunking, no duplication
            if full_text:
                yield full_text
            return

        except:
            continue

    yield "Unable to connect. Verify your OpenRouter API key in Streamlit Secrets."

# ═══════════════════════════════════════════════════════════════
# TOOLS
# ═══════════════════════════════════════════════════════════════
def web_search(query):
    if not CONFIG["SERPER_KEY"]: return ""
    def _s():
        r = requests.post("https://google.serper.dev/search", headers={"X-API-KEY":CONFIG["SERPER_KEY"],"Content-Type":"application/json"}, json={"q":query,"num":6}, timeout=3)
        return "\n\n".join([f"[{i+1}] {x['title']}\n{x['snippet']}" for i,x in enumerate(r.json().get("organic",[]))])
    result, err = web_search_cb.call(_s)
    return result if not err else ""

def _fetch_yahoo_batch(symbols):
    if not symbols: return {}
    try:
        r = requests.get("https://query1.finance.yahoo.com/v7/finance/quote", params={"symbols":",".join(symbols),"fields":"regularMarketPrice,regularMarketPreviousClose,currency"}, timeout=5, headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code != 200: return {}
        results = {}
        for item in r.json().get("quoteResponse",{}).get("result",[]):
            sym=item.get("symbol",""); pr=item.get("regularMarketPrice"); pv=item.get("regularMarketPreviousClose")
            if pr and pv and pr>0: results[sym]={"price":pr,"prev":pv,"change_pct":round(((pr-pv)/pv)*100,2),"currency":item.get("currency","USD")}
        return results
    except: return {}

@st.cache_data(ttl=30, show_spinner=False)
def get_live_prices():
    results = {}
    tickers = {
        "stocks":{"^GSPC":"S&P 500","^IXIC":"NASDAQ","^DJI":"Dow Jones","^FTSE":"FTSE 100","^N225":"Nikkei 225","AAPL":"Apple","MSFT":"Microsoft","NVDA":"Nvidia","TSLA":"Tesla","AMZN":"Amazon","GOOGL":"Alphabet","META":"Meta","JPM":"JPMorgan","GS":"Goldman Sachs"},
        "african_stocks":{"DANGCEM.LG":"Dangote Cement","MTNN.LG":"MTN Nigeria","GUARANTY.LG":"GTCO","ZENITHBANK.LG":"Zenith Bank","ACCESS.LG":"Access Holdings","^NGSEINDX":"NGX All-Share","NPN.JO":"Naspers","MTN.JO":"MTN Group","SBK.JO":"Standard Bank","FSR.JO":"FirstRand","SOL.JO":"Sasol","^JALSH":"JSE All-Share","MTNGH.GH":"MTN Ghana","^GSE":"GSE Composite","SNTS.BR":"Sonatel","^BRVM":"BRVM Composite","SCOM.NR":"Safaricom","EQTY.NR":"Equity Group","COMI.CA":"Commercial Int'l Bank","^CASE30":"EGX 30","ATW.CS":"Attijariwafa Bank","^MASI":"MASI"},
        "commodities":{"GC=F":"Gold","SI=F":"Silver","CL=F":"Crude Oil WTI","BZ=F":"Brent Crude","NG=F":"Natural Gas","HG=F":"Copper","PL=F":"Platinum","CC=F":"Cocoa","KC=F":"Coffee"},
        "forex":{"EURUSD=X":"EUR/USD","GBPUSD=X":"GBP/USD","USDJPY=X":"USD/JPY","USDCHF=X":"USD/CHF","AUDUSD=X":"AUD/USD","USDCAD=X":"USD/CAD","USDGHS=X":"USD/GHS (Cedi)","USDNGN=X":"USD/NGN (Naira)","USDZAR=X":"USD/ZAR (Rand)","USDKES=X":"USD/KES (Shilling)","USDEGP=X":"USD/EGP (Pound)","USDMAD=X":"USD/MAD (Dirham)","USDXOF=X":"USD/XOF (CFA Franc)","USDETB=X":"USD/ETB (Birr)","USDTZS=X":"USD/TZS (Shilling)","USDUGX=X":"USD/UGX (Shilling)"},
    }
    all_syms = [sym for grp in tickers.values() for sym in grp.keys()]
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = [ex.submit(_fetch_yahoo_batch, all_syms[i:i+20]) for i in range(0,len(all_syms),20)]
        yd = {}
        for f in as_completed(futures):
            try: yd.update(f.result())
            except: pass
    for grp,t in tickers.items():
        for sym,name in t.items():
            if sym in yd: d=yd[sym]; results[name]={"price":d["price"],"change_pct":d["change_pct"],"category":grp,"currency":d.get("currency","USD")}
    try:
        cids={"Bitcoin":"bitcoin","Ethereum":"ethereum","Solana":"solana","Cardano":"cardano","Ripple":"ripple","BNB":"binancecoin","USDT":"tether","USDC":"usd-coin"}
        r=requests.get(f"https://api.coingecko.com/api/v3/simple/price?ids={','.join(cids.values())}&vs_currencies=usd&include_24hr_change=true",timeout=8)
        if r.status_code==200:
            for n,cid in cids.items():
                coin=r.json().get(cid,{})
                if coin.get("usd"): results[n]={"price":coin["usd"],"change_pct":round(coin.get("usd_24h_change",0),2),"category":"crypto"}
    except: pass
    return results

@st.cache_data(ttl=300, show_spinner=False)
def fetch_financial_news():
    items=[]
    for url,src in [("https://feeds.content.dowjones.io/public/rss/mw_topstories","MarketWatch"),("https://finance.yahoo.com/news/rssindex","Yahoo Finance")]:
        try:
            r=requests.get(url,timeout=8,headers={"User-Agent":"Mozilla/5.0"})
            if r.status_code==200:
                for item in ET.fromstring(r.content).findall('.//item')[:5]:
                    t=item.find('title')
                    if t is not None and t.text and len(t.text)>10: items.append({"title":t.text.strip(),"source":src})
        except: continue
    seen=set(); uniq=[]
    for i in items:
        if i['title'] not in seen: seen.add(i['title']); uniq.append(i)
    return uniq[:10]

def render_price_row(name, data):
    c="#4ade80" if data["change_pct"]>=0 else "#f87171"; s="+" if data["change_pct"]>=0 else ""; a="▲" if data["change_pct"]>=0 else "▼"
    p=data["price"]
    ps=(f"${p:,.0f}" if p>=10000 else (f"${p:,.0f}" if p>=1000 else (f"${p:,.2f}" if p>=1 else f"${p:.4f}")))
    st.markdown(f'<div style="display:flex;justify-content:space-between;align-items:center;padding:0.3rem 0;font-size:0.75rem;border-bottom:1px solid rgba(255,255,255,0.04);"><span style="color:#8b949e;">{name}</span><span style="color:#e6edf3;font-weight:500;">{ps}</span><span style="color:{c};font-size:0.68rem;">{a} {s}{data["change_pct"]:.2f}%</span></div>', unsafe_allow_html=True)

def render_news_item(item):
    t=item["title"]; dt=t[:100]+"..." if len(t)>100 else t
    st.markdown(f'<div style="padding:0.35rem 0;border-bottom:1px solid rgba(255,255,255,0.04);"><div style="color:#e6edf3;font-size:0.73rem;line-height:1.4;">{dt}</div><div style="color:#484f58;font-size:0.63rem;">{item["source"]}</div></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TXID PAYMENT
# ═══════════════════════════════════════════════════════════════
def validate_txid_format(tx, cur):
    if not tx or not tx.strip(): return False
    tx=tx.strip()
    if cur=="BTC": return bool(re.match(r'^[a-fA-F0-9]{64}$',tx))
    if cur in ("ETH","USDC"): return bool(re.match(r'^0x[a-fA-F0-9]{64}$',tx))
    if cur=="SOL": return bool(re.match(r'^[1-9A-HJ-NP-Za-km-z]{87,88}$',tx))
    return False

def verify_crypto_payment(tx, cur, amt):
    tx=tx.strip()
    if not validate_txid_format(tx,cur): return False, f"Invalid {cur} TXID format."
    try:
        if cur=="BTC":
            for url in [f"https://blockchain.info/rawtx/{tx}",f"https://blockstream.info/api/tx/{tx}"]:
                try:
                    r=requests.get(url,timeout=10)
                    if r.status_code!=200: continue
                    outputs=r.json().get("out") or r.json().get("vout") or []
                    for out in outputs:
                        if (out.get("addr") or out.get("scriptpubkey_address",""))==CRYPTO_ADDRESSES["BTC"]:
                            v=out.get("value",0)
                            if v>1: v/=100_000_000
                            if abs(v-amt)<0.0001: return True, "Verified on Bitcoin."
                            return False, "Amount mismatch."
                    return False, "Wrong address."
                except: continue
            return False, "Could not verify."
        if cur in ("ETH","USDC"):
            ak=CONFIG.get("ETHERSCAN_API_KEY","")
            r=requests.get("https://api.etherscan.io/api",params={"module":"proxy","action":"eth_getTransactionByHash","txhash":tx,"apikey":ak},timeout=10)
            if r.status_code!=200: return False, "Could not connect."
            txd=r.json().get("result",{})
            if not txd: return False, "Not found."
            if txd.get("to","").lower()!=CRYPTO_ADDRESSES["ETH"].lower(): return False, "Wrong address."
            v=int(txd.get("value","0"),16)/1e18
            if abs(v-amt)<0.001: return True, "Verified."
            return False, "Amount mismatch."
        if cur=="SOL":
            r=requests.post("https://api.mainnet-beta.solana.com",json={"jsonrpc":"2.0","id":1,"method":"getTransaction","params":[tx,"json"]},headers={"Content-Type":"application/json"},timeout=10)
            if r.status_code!=200: return False, "Could not connect."
            d=r.json()
            if "error" in d: return False, "Not found."
            txd=d.get("result")
            if not txd: return False, "Not found."
            if txd.get("meta",{}).get("err"): return False, "Transaction failed."
            return True, "Verified on Solana."
    except Exception as e: return False, f"Error: {str(e)}"
    return False, "Unsupported."

def is_txid_previously_used(tx): return tx.strip() in st.session_state.get('verified_txids',[])
def mark_txid_as_used(tx):
    if 'verified_txids' not in st.session_state: st.session_state.verified_txids=[]
    st.session_state.verified_txids.append(tx.strip()); persist_current_state()

# ═══════════════════════════════════════════════════════════════
# VECTOR MEMORY
# ═══════════════════════════════════════════════════════════════
_EMBED_CLIENT=None; _EMBED_MODEL=None

def get_embedding(text):
    global _EMBED_CLIENT, _EMBED_MODEL
    if OPENAI_AVAILABLE and CONFIG.get("OPENAI_API_KEY"):
        if _EMBED_CLIENT is None:
            try: _EMBED_CLIENT=OpenAI(api_key=CONFIG["OPENAI_API_KEY"])
            except: _EMBED_CLIENT=False
        if _EMBED_CLIENT:
            try: return _EMBED_CLIENT.embeddings.create(model="text-embedding-3-small",input=text).data[0].embedding
            except: pass
    if LOCAL_EMBEDDING_AVAILABLE and _EMBED_MODEL is None:
        try: _EMBED_MODEL=SentenceTransformer('BAAI/bge-small-en-v1.5')
        except: _EMBED_MODEL=False
    if _EMBED_MODEL:
        try: return _EMBED_MODEL.encode(text,normalize_embeddings=True).tolist()
        except: pass
    return None

class DummyMemory:
    def search(self,q,k=3): return []
    def add_message(self,*a): pass

if FAISS_AVAILABLE:
    class VectorMemory:
        def __init__(self): self.index=None; self.metadata=[]; self._load_or_create()
        def _load_or_create(self):
            if os.path.exists(MEMORY_INDEX_PATH) and os.path.exists(MEMORY_META_PATH):
                try: self.index=faiss.read_index(MEMORY_INDEX_PATH); self.metadata=json.load(open(MEMORY_META_PATH)); return
                except: pass
            self.index=None; self.metadata=[]
        def _save(self):
            if self.index: faiss.write_index(self.index,MEMORY_INDEX_PATH)
            with open(MEMORY_META_PATH,'w') as f: json.dump(self.metadata,f,indent=2)
        def add_message(self,um,am,dom,acc):
            emb=get_embedding(um)
            if emb is None: return
            dim=len(emb)
            if self.index is None or self.index.d!=dim: self.index=faiss.IndexFlatIP(dim); self.metadata=[]
            emb_arr=np.array(emb,dtype=np.float32).reshape(1,-1); faiss.normalize_L2(emb_arr)
            self.metadata.append({"id":str(uuid.uuid4()),"timestamp":datetime.now().isoformat(),"domain":dom,"accuracy":acc,"content":f"User: {um}\nCAPITAN AI: {am}"})
            self.index.add(emb_arr); self._save()
        def search(self,q,k=3,sem_w=0.4,rec_w=0.3,acc_w=0.3):
            if self.index is None or self.index.ntotal==0: return []
            emb=get_embedding(q)
            if emb is None: return []
            dim=len(emb)
            if self.index.d!=dim: return []
            emb_arr=np.array(emb,dtype=np.float32).reshape(1,-1); faiss.normalize_L2(emb_arr)
            D,I=self.index.search(emb_arr,min(self.index.ntotal,20))
            cand=[]; now=datetime.now()
            for s,i in zip(D[0],I[0]):
                if i<0 or i>=len(self.metadata): continue
                m=self.metadata[i]; ss=(s+1)/2
                try: ts=datetime.fromisoformat(m["timestamp"])
                except: ts=now-timedelta(days=365)
                days_old=(now-ts).total_seconds()/86400.0
                recency=math.exp(-math.log(2)/CONFIG["MEMORY_DECAY_HALF_LIFE"]*days_old)
                accuracy=m.get("accuracy",3)/5.0
                final=sem_w*ss+rec_w*recency+acc_w*accuracy
                cand.append((final,m))
            cand.sort(key=lambda x:x[0],reverse=True)
            return [m["content"] for _,m in cand[:k]]
    memory_engine=VectorMemory()
else:
    memory_engine=DummyMemory()

# ═══════════════════════════════════════════════════════════════
# TOOL ROUTER
# ═══════════════════════════════════════════════════════════════
def decide_tools(query):
    try:
        r,err=llm_cb.call(call_llm,[{"role":"system","content":"Output only valid JSON arrays."},{"role":"user","content":f"Return JSON list from [web, prices, code, none]. Query: {query}"}],is_pro=False,use_specific_model="deepseek/deepseek-chat")
        if err: return ["none"]
        m=re.search(r'\[.*\]',r,re.DOTALL)
        if m:
            tools=json.loads(m.group())
            if isinstance(tools,list): return [t for t in tools if t in {"web","wolfram","code","prices","none"}]
    except: pass
    return ["none"]

# ═══════════════════════════════════════════════════════════════
# ELITE PROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════
def process_query(prompt, is_pro=False):
    domain=DomainRouter.classify(prompt)
    complexity=QueryComplexityAnalyzer.grade(prompt, domain)

    if domain=="trading_refuse":
        msgs=[{"role":"system","content":PERSONAS["trading_refuse"]},{"role":"user","content":prompt}]
        for chunk in call_llm_stream_fast(msgs, is_pro=is_pro): yield chunk
        return

    entities=entity_memory.extract_entities(prompt); entity_memory.add_entities(entities)
    mc=memory_engine.search(prompt, k=3)
    mt=""
    if mc: mt="RELEVANT MEMORY:\n"+"\n".join(f"  [{i+1}] {m}" for i,m in enumerate(mc))+"\n\n"
    et=entity_memory.get_summary(); gt=goal_tracker.get_context_for_ai()

    for pat in [r'\b(?:I (?:want|need|plan|aim|goal is) to\b[^.!?]+)',r'\b(?:my (?:goal|target|objective) is\b[^.!?]+)',r'\b(?:help me (?:prepare|study|learn|build|create|start|launch)\b[^.!?]+)']:
        m=re.search(pat, prompt, re.IGNORECASE)
        if m:
            gte=m.group(0).strip()
            if len(gte)>10: goal_tracker.add_goal(gte, domain); gt=goal_tracker.get_context_for_ai()
            break

    tc=""
    if is_pro:
        tools=decide_tools(prompt)
        if "web" in tools and CONFIG["SERPER_KEY"]:
            sr=web_search(prompt[:150])
            if sr: tc+="\nWEB SEARCH RESULTS:\n"+sr+"\n"
        if "prices" in tools:
            prices=get_live_prices()
            if prices:
                lines=[f"{n}: ${d['price']:,.2f} ({'+' if d['change_pct']>=0 else ''}{d['change_pct']:.2f}%)" for n,d in prices.items()]
                tc+="\nLIVE PRICES:\n"+"\n".join(lines)+"\n"
        if domain in ("finance","african_finance","macro","quant"):
            news=fetch_financial_news()
            if news:
                hl=[f"[{i+1}] {n['title']} — {n['source']}" for i,n in enumerate(news[:5])]
                tc+="\nMARKET NEWS:\n"+"\n".join(hl)+"\n"

    elite_scaffold=None
    if is_pro and complexity in ("standard","deep") and len(prompt.split())>20:
        elite_scaffold=EliteReasoningEngine.decompose(prompt, domain, complexity, is_pro)

    persona=PERSONAS.get(domain, PERSONAS["general"])
    ctx_blocks=[]
    if elite_scaffold: ctx_blocks.append(EliteReasoningEngine.build_scaffold_context(elite_scaffold))
    if gt: ctx_blocks.append(gt)
    if et: ctx_blocks.append(et)
    if mt: ctx_blocks.append(mt)
    if ctx_blocks: persona="\n\n".join(ctx_blocks)+"\n\n"+persona
    if tc: persona+="\n\n=== LIVE INTELLIGENCE ===\n"+tc+"=== END LIVE INTELLIGENCE ===\n"

    emotional_patterns=[r'\b(tired|sad|lonely|stressed|anxious|worried|overwhelmed|depressed|upset|heartbroken|grieving)\b',r'\b(i\'m feeling|i feel|i am feeling|feeling kinda|feeling a bit|been feeling)\b',r'\b(hard day|rough day|tough week|difficult time|struggling)\b']
    if any(re.search(p, prompt, re.IGNORECASE) for p in emotional_patterns):
        persona+="\n\nEMOTIONAL CONTEXT: Acknowledge briefly and genuinely. Offer practical support. Do not over-elaborate."

    word_count=len(prompt.split())
    if word_count<8 or "briefly" in prompt.lower() or "concise" in prompt.lower(): persona+="\n\nBE CONCISE."
    elif "detailed" in prompt.lower() or "comprehensive" in prompt.lower() or complexity=="deep": persona+="\n\nBE THOROUGH."

    model_override=None
    if complexity=="deep" and is_pro: model_override=CONFIG["DEEP_MODEL"]
    elif complexity=="simple" and not is_pro: model_override=CONFIG["FAST_MODEL"]

    messages=[{"role":"system","content":persona},{"role":"user","content":prompt}]
    fr=""
    for chunk in call_llm_stream_fast(messages, is_pro=is_pro, model_override=model_override):
        fr=chunk  # Use the chunk directly — it's now the complete text
        yield fr

    if is_pro and complexity=="deep" and elite_scaffold and len(fr)>300:
        try:
            critique_data=EliteReasoningEngine.critique(prompt, fr, is_pro)
            if critique_data and critique_data.get("verdict") in ("WEAK","ACCEPTABLE"):
                scaffold_text=EliteReasoningEngine.build_scaffold_context(elite_scaffold)
                refined=EliteReasoningEngine.synthesize_elite(prompt, scaffold_text, critique_data, fr, is_pro)
                if refined and refined!=fr:
                    sep="\n\n---\n*✦ Elite Refined Response (adversarial critique applied):*\n\n"
                    fr=fr+sep+refined
                    yield sep+refined
        except: pass

    try:
        scores=EliteSelfEvaluator.evaluate(prompt, fr, domain, is_pro) if is_pro else {"accuracy":3.0}
        acc=scores.get("accuracy",3.0) if isinstance(scores,dict) else 3.0
    except: acc=3.0
    memory_engine.add_message(prompt, fr, domain, acc)

# ═══════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════
st.set_page_config(page_title="CAPITAN AI", page_icon="⚓", layout="centered", initial_sidebar_state="expanded")

st.markdown('<link rel="manifest" href="/.streamlit/static/manifest.json">', unsafe_allow_html=True)
st.markdown('<meta name="theme-color" content="#0d1117">', unsafe_allow_html=True)
st.markdown('<meta name="mobile-web-app-capable" content="yes">', unsafe_allow_html=True)
st.markdown('<meta name="apple-mobile-web-app-capable" content="yes">', unsafe_allow_html=True)
st.markdown('<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">', unsafe_allow_html=True)
st.markdown('<meta name="apple-mobile-web-app-title" content="CAPITAN AI">', unsafe_allow_html=True)

st.components.v1.html("""
<!DOCTYPE html><html><head>
<style>body{margin:0;padding:0;}#cap-install-btn{position:fixed;bottom:0;left:0;right:0;background:#161b22;border-top:2px solid #4ade80;padding:14px 18px;display:flex;align-items:center;justify-content:space-between;z-index:99999;font-family:Inter,system-ui,sans-serif;animation:slideUp .35s ease-out;box-shadow:0 -8px 32px rgba(0,0,0,.6)}#cap-install-btn.hidden{display:none}.install-btn{background:#4ade80;color:#000;border:none;padding:10px 22px;border-radius:50px;font-size:14px;font-weight:700;cursor:pointer;white-space:nowrap}.dismiss-btn{background:transparent;color:#484f58;border:none;font-size:20px;cursor:pointer;padding:4px 8px}@keyframes slideUp{from{transform:translateY(100%);opacity:0}to{transform:translateY(0);opacity:1}}</style></head><body>
<div id="cap-install-btn" class="hidden"><div style="display:flex;align-items:center;gap:14px"><div style="width:48px;height:48px;background:#0d1117;border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:24px;border:1px solid #30363d">⚓</div><div><div style="color:#e6edf3;font-size:15px;font-weight:700">CAPITAN AI</div><div style="color:#4ade80;font-size:12px;margin-top:2px">Install · Free · Works offline</div></div></div><div style="display:flex;align-items:center;gap:8px"><button class="install-btn" onclick="installApp()">Install</button><button class="dismiss-btn" onclick="dismissBanner()">✕</button></div></div>
<script>var dp=null,isSA=window.matchMedia('(display-mode:standalone)').matches||window.navigator.standalone===true;window.addEventListener('beforeinstallprompt',function(e){e.preventDefault();dp=e;if(!isSA)document.getElementById('cap-install-btn').classList.remove('hidden')});function installApp(){if(dp){dp.prompt();dp.userChoice.then(function(r){document.getElementById('cap-install-btn').classList.add('hidden');dp=null})}else if(isSA)alert('Already installed!');else alert('Look for the install icon (⊕) in your address bar.')}function dismissBanner(){document.getElementById('cap-install-btn').classList.add('hidden')}</script></body></html>
""", height=80)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root{{--bg-primary:#0d1117;--bg-secondary:#161b22;--bg-tertiary:#21262d;--border:#30363d;--text-primary:#e6edf3;--text-secondary:#8b949e;--text-muted:#484f58;--accent:#4ade80;--accent-dim:rgba(74,222,128,0.15);--africa-gold:#f0c040;--red:#f87171;--radius:12px;--radius-sm:8px;--font-xs:0.65rem;--font-sm:0.72rem;--font-md:0.8rem;--font-lg:0.9rem;--font-xl:1.1rem;--font-2xl:1.5rem}}
.stApp{{background:var(--bg-primary)!important;color:var(--text-primary)!important;font-family:'Inter',sans-serif;font-size:var(--font-sm)!important}}
.main .block-container{{padding:1rem 1rem 0 1rem!important;max-width:800px!important;font-size:var(--font-sm)!important}}
section[data-testid="stSidebar"]{{background:var(--bg-secondary)!important;border-right:1px solid var(--border)!important;font-size:var(--font-xs)!important}}
section[data-testid="stSidebar"] .stButton button{{background:transparent;border:none;color:var(--text-secondary);padding:0.4rem 0.6rem;font-size:var(--font-xs);text-align:left;border-radius:var(--radius-sm);width:100%;transition:all 0.2s}}
section[data-testid="stSidebar"] .stButton button:hover{{background:var(--bg-tertiary);color:var(--text-primary)}}
section[data-testid="stSidebar"] p,section[data-testid="stSidebar"] span,section[data-testid="stSidebar"] div,section[data-testid="stSidebar"] label,section[data-testid="stSidebar"] .stCaption{{font-size:var(--font-xs)!important}}
section[data-testid="stSidebar"] .stExpander summary,section[data-testid="stSidebar"] .stExpander p{{font-size:var(--font-xs)!important}}
.chat-message{{padding:0.85rem 1.1rem;margin:0.4rem 0;line-height:1.55;font-size:var(--font-sm)}}
.chat-user{{background:var(--bg-tertiary);border-radius:var(--radius);margin-left:auto;max-width:85%;color:var(--text-primary)}}
.chat-assistant{{background:transparent;border-left:2px solid var(--accent);border-radius:0 var(--radius-sm) var(--radius-sm) 0;padding-left:1rem;max-width:95%;color:var(--text-primary)}}
.chat-assistant code{{background:var(--bg-tertiary);color:var(--accent);padding:0.12rem 0.35rem;border-radius:3px;font-family:'JetBrains Mono',monospace;font-size:0.78em}}
.chat-assistant pre{{background:var(--bg-secondary);border:1px solid var(--border);padding:0.85rem;border-radius:var(--radius-sm);overflow-x:auto;font-size:var(--font-xs)}}
.welcome-container{{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:40vh;text-align:center;padding:1.5rem}}
.welcome-title{{font-size:var(--font-2xl);font-weight:600;color:var(--text-primary);margin-bottom:0.4rem}}
.welcome-subtitle{{font-size:var(--font-sm);color:var(--text-secondary);margin-bottom:1.5rem}}
.stChatInput textarea{{background:var(--bg-secondary)!important;border:1px solid var(--border)!important;color:var(--text-primary)!important;border-radius:var(--radius)!important;padding:0.65rem 0.85rem!important;font-size:var(--font-xs)!important}}
.stChatInput textarea:focus{{border-color:var(--accent)!important;box-shadow:0 0 0 3px var(--accent-dim)!important}}
.thinking-indicator{{display:flex;align-items:center;gap:0.6rem;padding:0.85rem;color:var(--text-muted);font-size:var(--font-xs)}}
.thinking-dots{{display:flex;gap:3px}}
.thinking-dot{{width:5px;height:5px;border-radius:50%;background:var(--accent);animation:dotPulse 1.4s ease-in-out infinite}}
.thinking-dot:nth-child(2){{animation-delay:0.2s}}.thinking-dot:nth-child(3){{animation-delay:0.4s}}
@keyframes dotPulse{{0%,80%,100%{{opacity:0.3;transform:scale(0.8)}}40%{{opacity:1;transform:scale(1.2)}}}}
.status-bar{{display:flex;align-items:center;justify-content:center;gap:0.4rem;padding:0.4rem;font-size:var(--font-xs);color:var(--text-muted);border-top:1px solid var(--border);margin-top:0.75rem}}
.status-dot{{width:6px;height:6px;border-radius:50%;background:var(--accent)}}
.ai-note{{font-size:0.6rem;color:var(--text-muted);padding-left:1rem;margin-top:-0.2rem;margin-bottom:0.4rem}}
.nav-label{{font-size:0.6rem;text-transform:uppercase;letter-spacing:0.05em;color:var(--text-muted);padding:0.4rem 0.6rem 0.2rem}}
.section-header{{font-size:0.58rem;color:#484f58;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.2rem}}
.category-header{{font-size:0.62rem;color:#4ade80;font-weight:500;padding:0.4rem 0 0.2rem 0;margin-top:0.2rem}}
.africa-header{{font-size:0.62rem;color:var(--africa-gold);font-weight:500;padding:0.4rem 0 0.2rem 0;margin-top:0.2rem}}
hr{{border-color:var(--border)!important;margin:0.6rem 0!important}}
.upgrade-section{{background:var(--bg-tertiary);border:1px solid var(--accent);border-radius:var(--radius);padding:0.85rem;margin-top:0.4rem;font-size:var(--font-xs)}}
.upgrade-section h4{{font-size:var(--font-sm)}}
.crypto-address{{background:var(--bg-primary);padding:0.4rem;border-radius:3px;font-family:'JetBrains Mono',monospace;font-size:0.65rem;word-break:break-all;color:var(--text-secondary)}}
.goal-item{{display:flex;justify-content:space-between;align-items:center;padding:0.3rem 0;font-size:0.7rem;border-bottom:1px solid rgba(255,255,255,0.04)}}
.privacy-badge{{text-align:center;padding:0.6rem;margin-top:0.4rem;border-top:1px solid var(--border)}}
.privacy-badge-text{{font-size:0.58rem;color:var(--text-muted);line-height:1.4}}
.privacy-badge-text strong{{color:var(--accent)}}
.free-limit-bar{{margin:0.4rem 0.6rem}}
.free-limit-bar-inner{{font-size:0.6rem;color:var(--text-muted);text-align:center;margin-bottom:0.2rem}}
.free-limit-progress{{background:rgba(255,255,255,0.05);height:2px;border-radius:1px;overflow:hidden}}
.free-limit-fill{{background:var(--accent);height:100%;border-radius:1px;transition:width 0.3s}}
@media(max-width:768px){{.chat-user,.chat-assistant{{max-width:100%}}.welcome-title{{font-size:var(--font-xl)}}}}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# SESSION STATE — Pro/Founder status ALWAYS loaded from disk
# ═══════════════════════════════════════════════════════════════
lm, lp, lf, lch, ltx, lpr, ldc, ldr = load_persistent_state()
lproj = load_projects()

st.session_state.messages = lm
st.session_state.is_pro = lp
st.session_state.is_founder = lf
st.session_state.chat_history = lch
st.session_state.verified_txids = ltx
st.session_state.user_preferences = lpr
st.session_state.projects = lproj
st.session_state.goals = load_goals()

if 'current_chat_id' not in st.session_state: st.session_state.current_chat_id = str(uuid.uuid4())
if 'current_project_id' not in st.session_state: st.session_state.current_project_id = None
if 'model' not in st.session_state: st.session_state.model = "smart"
if 'web_search_enabled' not in st.session_state: st.session_state.web_search_enabled = True
if 'show_upgrade' not in st.session_state: st.session_state.show_upgrade = False
st.session_state.daily_count = ldc
st.session_state.daily_reset = ldr

FREE_DAILY_LIMIT = CONFIG["FREE_DAILY_LIMIT"]
try:
    reset_time = datetime.fromisoformat(st.session_state.daily_reset)
except (ValueError, TypeError):
    reset_time = datetime.now()
    st.session_state.daily_reset = reset_time.isoformat()

if datetime.now() - reset_time > timedelta(hours=24):
    st.session_state.daily_count = 0
    st.session_state.daily_reset = datetime.now().isoformat()

remaining_free = max(0, FREE_DAILY_LIMIT - st.session_state.daily_count)

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f'<div style="display:flex;align-items:center;gap:0.6rem;padding:0.4rem 0.6rem"><img src="{CAPITAN_LOGO_BASE64}" width="30" height="30"><span style="font-size:0.95rem;font-weight:600;color:#e6edf3">CAPITAN AI</span><span style="font-size:0.55rem;color:#f0c040;margin-left:auto">SOVEREIGN</span></div>', unsafe_allow_html=True)

    if not st.session_state.is_pro and not st.session_state.is_founder:
        pct_used = (st.session_state.daily_count / FREE_DAILY_LIMIT) * 100
        bar_color = "#4ade80" if pct_used < 70 else ("#f0c040" if pct_used < 90 else "#f87171")
        st.markdown(f'<div class="free-limit-bar"><div class="free-limit-bar-inner">{remaining_free}/{FREE_DAILY_LIMIT} messages today</div><div class="free-limit-progress"><div class="free-limit-fill" style="width:{pct_used}%;background:{bar_color}"></div></div></div>', unsafe_allow_html=True)

    if st.button("+ New Chat", use_container_width=True, key="nc"):
        if st.session_state.messages: st.session_state.chat_history.append({"id":st.session_state.current_chat_id,"title":st.session_state.messages[0]["content"][:50] if st.session_state.messages else "New Chat","messages":st.session_state.messages.copy(),"timestamp":datetime.now().isoformat(),"project_id":st.session_state.current_project_id})
        st.session_state.messages = []; st.session_state.current_chat_id = str(uuid.uuid4()); persist_current_state(); st.rerun()

    st.markdown("---")
    with st.expander("📁 Projects", expanded=False):
        if st.button("+ New Project", use_container_width=True, key="np"):
            npid = str(uuid.uuid4()); st.session_state.projects[npid] = {"id":npid,"name":"Untitled Project","created":datetime.now().isoformat(),"chats":[],"files":[]}
            st.session_state.current_project_id = npid; persist_current_state(); st.rerun()
        for pid, proj in st.session_state.projects.items():
            c1, c2 = st.columns([3,1])
            with c1:
                if st.button(f"{'📌' if pid==st.session_state.current_project_id else '📁'} {proj['name'][:25]}",use_container_width=True,key=f"pj_{pid}"): st.session_state.current_project_id = pid; persist_current_state(); st.rerun()
            with c2:
                if st.button("✕",key=f"dp_{pid}"): del st.session_state.projects[pid]; st.session_state.current_project_id = None if st.session_state.current_project_id==pid else st.session_state.current_project_id; persist_current_state(); st.rerun()

    st.markdown('<div class="nav-label">Chats</div>', unsafe_allow_html=True)
    for chat in reversed(st.session_state.chat_history[-5:]):
        if st.button(chat.get("title","Untitled")[:30],use_container_width=True,key=f"ch_{chat['id']}"): st.session_state.messages = chat["messages"]; st.session_state.current_chat_id = chat["id"]; st.session_state.current_project_id = chat.get("project_id"); persist_current_state(); st.rerun()

    with st.expander("🎯 Goals", expanded=False):
        ag = goal_tracker.get_active_goals()
        if ag:
            for g in ag[:5]:
                da = (datetime.now()-datetime.fromisoformat(g["created"])).days
                st.markdown(f'<div class="goal-item"><span style="color:#e6edf3;font-size:0.7rem">{g["description"][:40]}...</span><span style="color:#484f58;font-size:0.6rem">{da}d ago</span></div>',unsafe_allow_html=True)
        else: st.caption("Goals detected from conversations.")

    st.markdown("---"); st.markdown('<div class="nav-label">Files</div>', unsafe_allow_html=True)
    uf = st.file_uploader("Upload",type=["pdf","txt","py","csv","json","png","jpg"],label_visibility="collapsed",key="fu")
    if uf:
        st.success(f"Uploaded: {uf.name}")
        if st.session_state.current_project_id: st.session_state.projects[st.session_state.current_project_id]["files"].append({"name":uf.name,"timestamp":datetime.now().isoformat()}); persist_current_state()

    with st.expander("🧠 Memory", expanded=False):
        es = entity_memory.get_summary()
        if es: st.markdown(f'<div style="font-size:0.7rem;color:#e6edf3">{es}</div>',unsafe_allow_html=True)
        else: st.caption("Remembers people, companies, projects.")

    st.markdown("---")
    with st.expander("⚙️ Settings", expanded=False):
        model = st.selectbox("Model",["CAPITAN Fast","CAPITAN Smart","CAPITAN Deep Think"],label_visibility="collapsed",key="ms")
        st.session_state.model = {"CAPITAN Fast":"fast","CAPITAN Smart":"smart","CAPITAN Deep Think":"deep"}.get(model,"smart")
        st.session_state.web_search_enabled = st.toggle("Web Search",value=st.session_state.web_search_enabled)

    st.markdown("---")
    if st.session_state.is_pro or st.session_state.is_founder:
        with st.expander("📈 Live Prices", expanded=False):
            prices = get_live_prices()
            if prices:
                cats = {"Global Markets":[],"🟡 African Markets":[],"Crypto":[],"Commodities":[],"Forex & African FX":[]}
                for n,d in prices.items():
                    cat = d.get("category","stocks")
                    if cat=="stocks": cats["Global Markets"].append((n,d))
                    elif cat=="african_stocks": cats["🟡 African Markets"].append((n,d))
                    elif cat=="crypto": cats["Crypto"].append((n,d))
                    elif cat=="commodities": cats["Commodities"].append((n,d))
                    elif cat=="forex": cats["Forex & African FX"].append((n,d))
                for cn,items in cats.items():
                    if items:
                        hc = "africa-header" if "African" in cn else "category-header"
                        st.markdown(f'<div class="{hc}">{cn}</div>',unsafe_allow_html=True)
                        for n,d in items: render_price_row(n,d)
                st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")
                if st.button("🔄 Refresh",use_container_width=True,key="rp"): get_live_prices.clear(); st.rerun()
            else: st.caption("Loading...")
        with st.expander("📰 Market News", expanded=False):
            news = fetch_financial_news()
            if news:
                st.markdown('<div class="section-header">Latest Headlines</div>',unsafe_allow_html=True)
                for i in news[:5]: render_news_item(i)
                st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")
            else: st.caption("No news available")

    st.markdown("---")
    if st.session_state.is_founder: st.markdown('<div style="background:#4ade80;color:#000;padding:0.2rem 0.6rem;border-radius:20px;font-size:0.7rem;font-weight:600;text-align:center">⚓ FOUNDER</div>',unsafe_allow_html=True)
    elif st.session_state.is_pro: st.markdown('<div style="background:#4ade80;color:#000;padding:0.2rem 0.6rem;border-radius:20px;font-size:0.7rem;font-weight:600;text-align:center">◆ PRO</div>',unsafe_allow_html=True)
    else:
        if st.button("✨ Upgrade to Pro — $15/mo",use_container_width=True,key="ub"): st.session_state.show_upgrade = not st.session_state.show_upgrade

    if st.session_state.show_upgrade and not st.session_state.is_pro and not st.session_state.is_founder:
        st.markdown('<div class="upgrade-section">',unsafe_allow_html=True); st.markdown("#### 🚀 Unlock Pro")
        for f in ["Unlimited messages","Live market prices & news","African stock data","Web search","Multi-model intelligence","Vector memory","Elite adversarial reasoning","Project management","Goal tracking","Priority support"]: st.markdown(f"• {f}")
        st.markdown("---"); st.markdown("**💳 Pay with crypto:**")
        crypto = st.selectbox("Currency",["BTC","ETH","USDC","SOL"],key="cs",format_func=lambda x:f"{x} — {PRO_PRICE_CRYPTO.get(x,0)} {x}")
        price = PRO_PRICE_CRYPTO.get(crypto,15)
        st.markdown(f'<div style="background:var(--bg-primary);border-radius:8px;padding:0.75rem;margin:0.5rem 0"><div style="display:flex;justify-content:space-between;margin-bottom:0.5rem"><span style="color:#8b949e;font-size:0.75rem">Amount</span><span style="color:#4ade80;font-weight:600;font-size:0.8rem">{price} {crypto}</span></div><div style="display:flex;justify-content:space-between"><span style="color:#8b949e;font-size:0.75rem">USD Value</span><span style="color:#e6edf3;font-size:0.8rem">~${PRO_PRICE_USD}</span></div><div style="margin-top:0.5rem"><span style="color:#8b949e;font-size:0.7rem">Send to:</span></div><div class="crypto-address" style="margin-top:0.25rem">{CRYPTO_ADDRESSES.get(crypto,"")}</div></div>',unsafe_allow_html=True)
        if st.button("📋 Copy",use_container_width=True,key="ca"): st.toast("Copied!")
        st.markdown("---"); st.markdown("**🔍 Verify with TXID:**")
        tx = st.text_input("Transaction Hash",placeholder=f"Paste {crypto} TXID...",key="ti")
        if tx and validate_txid_format(tx,crypto): st.markdown(f'<a href="{EXPLORER_LINKS.get(crypto,"")}{tx}" target="_blank" style="color:#4ade80;font-size:0.75rem">🔗 Explorer</a>',unsafe_allow_html=True)
        c1,c2 = st.columns([2,1])
        with c1:
            if st.button("🔍 Verify & Activate",use_container_width=True,key="vp"):
                if not tx: st.error("Enter TXID.")
                elif is_txid_previously_used(tx): st.warning("Already used.")
                else:
                    with st.spinner("Verifying..."):
                        v,msg = verify_crypto_payment(tx,crypto,price)
                        if v: mark_txid_as_used(tx); st.session_state.is_pro=True; st.session_state.messages=[]; st.session_state.show_upgrade=False; persist_current_state(); st.success(f"✅ {msg}"); st.balloons(); time.sleep(1.5); st.rerun()
                        else: st.error(f"❌ {msg}")
        with c2:
            if st.button("Clear",use_container_width=True,key="ct"): st.rerun()
        st.markdown("**🔑 Pro key:**")
        key = st.text_input("Key",type="password",placeholder="cap-pro-...",key="pk")
        if st.button("Activate Key",use_container_width=True,key="ak"):
            if key==CONFIG.get("FOUNDER_KEY",""): st.session_state.is_founder=True; st.session_state.is_pro=True; st.session_state.messages=[]; st.session_state.show_upgrade=False; persist_current_state(); st.success("Founder!"); st.rerun()
            elif key.startswith("cap-pro-"): st.session_state.is_pro=True; st.session_state.messages=[]; st.session_state.show_upgrade=False; persist_current_state(); st.success("Pro!"); st.rerun()
            else: st.error("Invalid.")
        st.markdown('</div>',unsafe_allow_html=True)

    if st.session_state.is_founder:
        if st.button("Exit Founder",use_container_width=True): st.session_state.is_founder=False; st.session_state.is_pro=False; st.session_state.messages=[]; persist_current_state(); st.rerun()
    elif st.session_state.is_pro:
        if st.button("Disconnect Pro",use_container_width=True): st.session_state.is_pro=False; st.session_state.messages=[]; persist_current_state(); st.rerun()

    st.markdown('<div class="privacy-badge"><div class="privacy-badge-text">🔒 <strong>Privacy First</strong> — No accounts. No tracking.<br>No personal data stored. TXID is your receipt.<br>Built by <strong>Sovereign AI Technologies</strong><br>Osinachi Chukwu · 2026</div></div>',unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════
if not st.session_state.messages:
    st.markdown(f'<div class="welcome-container"><div class="welcome-logo"><img src="{CAPITAN_LOGO_BASE64}" width="56" height="56" alt="CAPITAN AI"></div><div class="welcome-title">How can I help today?</div><div class="welcome-subtitle">World-class intelligence. African market depth. Zero-cost architecture.</div></div>',unsafe_allow_html=True)
    suggestions = [("📊 African Markets","Analyze the NGX All-Share and key Nigerian banking stocks"),("💻 Write Code","Write a Python backtesting framework for trading strategies"),("💰 Investment Memo","Write an investment memo on MTN Group with African market context"),("🌍 Macro Analysis","Analyze AfCFTA impact on cross-border payments in West Africa")]
    cols = st.columns(2)
    for i,(l,q) in enumerate(suggestions):
        with cols[i%2]:
            if st.button(l,use_container_width=True,key=f"s_{i}"): st.session_state.messages.append({"role":"user","content":q,"id":str(uuid.uuid4())}); persist_current_state(); st.rerun()

for msg in st.session_state.messages:
    if msg["role"]=="user": st.markdown(f'<div class="chat-message chat-user">{msg["content"]}</div>',unsafe_allow_html=True)
    else: st.markdown(f'<div class="chat-message chat-assistant">{msg["content"]}</div>',unsafe_allow_html=True); st.markdown('<div class="ai-note">CAPITAN AI can make mistakes. Verify important information.</div>',unsafe_allow_html=True)

ip = st.session_state.is_pro or st.session_state.is_founder
ml = {"fast":"CAPITAN Fast","smart":"CAPITAN Smart","deep":"CAPITAN Deep Think"}
cml = ml.get(st.session_state.model,"CAPITAN Smart")
st.markdown(f'<div class="status-bar"><div class="status-dot"></div>{cml}{" · Web" if st.session_state.web_search_enabled else ""}{" · PRO" if ip else " · Free"}{" · "+st.session_state.projects[st.session_state.current_project_id]["name"][:20] if st.session_state.current_project_id else ""}</div>',unsafe_allow_html=True)

prompt = st.chat_input("Ask CAPITAN AI anything...")

if prompt:
    ip = st.session_state.is_pro or st.session_state.is_founder
    if not ip:
        if remaining_free <= 0:
            st.warning("100 messages used today. Resets in 24h. Upgrade to Pro for unlimited.")
            if st.button("Upgrade to Pro — $15/month"): st.session_state.show_upgrade = True
            st.stop()
        st.session_state.daily_count += 1

    st.session_state.messages.append({"role":"user","content":prompt,"id":str(uuid.uuid4())}); persist_current_state()
    st.markdown(f'<div class="chat-message chat-user">{prompt}</div>',unsafe_allow_html=True)

    tp = st.empty()
    for stage in ["Thinking...","Analyzing...","Generating..."]:
        tp.markdown(f'<div class="thinking-indicator"><div class="thinking-dots"><div class="thinking-dot"></div><div class="thinking-dot"></div><div class="thinking-dot"></div></div>{stage}</div>',unsafe_allow_html=True)
        time.sleep(0.18)

    rp = st.empty(); fr = ""
    for chunk in process_query(prompt, ip):
        fr = chunk
        rp.markdown(f'<div class="chat-message chat-assistant">{fr}</div>',unsafe_allow_html=True)

    tp.empty()
    st.markdown('<div class="ai-note">CAPITAN AI can make mistakes. Verify important information.</div>',unsafe_allow_html=True)

    st.session_state.messages.append({"role":"assistant","content":fr,"id":str(uuid.uuid4())}); persist_current_state()
    st.rerun()