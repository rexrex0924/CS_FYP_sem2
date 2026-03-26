import streamlit as st
import torch
import re
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 1. Constants and Prompts (Same as your original) ---
PROMPT_TEMPLATE = """Question: {question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n\n{agent_prompt}\n\nEnd your response by clearly stating your final answer as exactly one letter (A, B, C, or D) inside <answer> tags, like this: <answer>A</answer>"""

DEBATE_PROMPT_TEMPLATE = """Question: {question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n\nYou previously chose {my_ans}. Another agent chose {other_ans} with this reasoning:\n"{other_reasoning}"\n\nCritique their reasoning. Do you concede to {other_ans}, or hold your ground on {my_ans}?\nEnd your response by clearly stating your final choice as exactly one letter ({my_ans} or {other_ans}) inside <answer> tags, like this: <answer>{my_ans}</answer>"""

AGENT_PROMPTS = {
    1: "Solve this step-by-step using logical deduction.",
    2: "Identify the most common trap or misconception in this question and avoid it.",
    3: "Give your immediate, most confident answer based on core principles."
}

# --- 2. Model Loading (Cached so it only loads once) ---
@st.cache_resource
def load_model(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model, device

def generate_text(tokenizer, model, device, prompt, max_new_tokens=512):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=0.7, 
            do_sample=True, pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def extract_answer_and_reasoning(response_text: str):
    match = re.search(r'<answer>\s*([A-D])\s*</answer>', response_text, re.IGNORECASE)
    if match:
        return match.group(1).upper(), response_text[:match.start()].strip()
    matches = re.findall(r'\b([A-D])\b', response_text.upper())
    return (matches[-1], response_text.strip()) if matches else ("", response_text.strip())

# --- 3. Streamlit UI Layout ---
st.set_page_config(page_title="MAD-Graph Visualization", layout="wide")
st.title("MAD-Graph: LLM Positional Bias Mitigation")
st.markdown("Visualizing the Multi-Agent Debate Process for Multiple Choice Questions.")

# Sidebar for Model Configuration
with st.sidebar:
    st.header("Settings")
    model_name = st.text_input("Hugging Face Model Path", "google/gemma-2b-it")
    if st.button("Load Model"):
        with st.spinner("Loading model into GPU... This may take a minute."):
            st.session_state['tokenizer'], st.session_state['model'], st.session_state['device'] = load_model(model_name)
        st.success("Model Loaded!")

# Main UI for Input
with st.form("mcq_form"):
    question = st.text_area("Question", "What is the capital of France?")
    col1, col2 = st.columns(2)
    with col1:
        opt_A = st.text_input("Option A", "Berlin")
        opt_B = st.text_input("Option B", "Madrid")
    with col2:
        opt_C = st.text_input("Option C", "Paris")
        opt_D = st.text_input("Option D", "Rome")
    
    submitted = st.form_submit_button("Run MAD-Graph Generation")

# --- 4. Execution and Visualization ---
if submitted:
    if 'model' not in st.session_state:
        st.error("Please load the model from the sidebar first!")
        st.stop()
        
    tokenizer = st.session_state['tokenizer']
    model = st.session_state['model']
    device = st.session_state['device']
    
    st.divider()
    
    # Phase 1: Divergent Generation
    st.header("Phase 1: Divergent Thinking")
    cols = st.columns(3)
    
    initial_responses = {}
    
    with st.spinner("Agents are thinking..."):
        for agent_id, agent_prompt in AGENT_PROMPTS.items():
            prompt = PROMPT_TEMPLATE.format(question=question, A=opt_A, B=opt_B, C=opt_C, D=opt_D, agent_prompt=agent_prompt)
            raw_response = generate_text(tokenizer, model, device, prompt)
            ans, reasoning = extract_answer_and_reasoning(raw_response)
            initial_responses[agent_id] = {"ans": ans, "reasoning": reasoning}
            
            with cols[agent_id - 1]:
                st.subheader(f"Agent {agent_id}")
                st.info(f"**Persona:** {agent_prompt}")
                st.markdown(f"**Initial Answer:** `{ans}`")
                with st.expander("Show Reasoning"):
                    st.write(reasoning)

    valid_votes = {aid: r["ans"] for aid, r in initial_responses.items() if r["ans"] in ["A", "B", "C", "D"]}
    
    if len(set(valid_votes.values())) == 1:
        st.success("All agents agreed! No debate needed.")
        final_answer = list(valid_votes.values())[0]
        
    else:
        # Phase 2: Debate
        st.header("Phase 2: Cross-Critique Debate")
        final_votes = dict(valid_votes)
        
        debate_cols = st.columns(len(valid_votes))
        
        with st.spinner("Agents are debating..."):
            col_idx = 0
            for agent_id, my_ans in valid_votes.items():
                disagreeing = [(aid, a) for aid, a in valid_votes.items() if a != my_ans]
                if disagreeing:
                    other_agent_id, other_ans = disagreeing[0]
                    other_reasoning = initial_responses[other_agent_id]["reasoning"]
                    
                    debate_prompt = DEBATE_PROMPT_TEMPLATE.format(
                        question=question, A=opt_A, B=opt_B, C=opt_C, D=opt_D,
                        my_ans=my_ans, other_ans=other_ans, other_reasoning=other_reasoning[:800]
                    )
                    
                    raw_debate = generate_text(tokenizer, model, device, debate_prompt)
                    new_ans, debate_reasoning = extract_answer_and_reasoning(raw_debate)
                    if new_ans in [my_ans, other_ans]:
                        final_votes[agent_id] = new_ans
                        
                    with debate_cols[col_idx]:
                        st.subheader(f"Agent {agent_id}'s Turn")
                        st.warning(f"Critiquing Agent {other_agent_id} (who chose {other_ans})")
                        st.markdown(f"**New Answer:** `{new_ans}`")
                        with st.expander("Show Debate Reasoning"):
                            st.write(debate_reasoning)
                col_idx += 1
                
        final_answer = Counter(final_votes.values()).most_common(1)[0][0]

    # Phase 3: Result
    st.divider()
    st.header(f"🏆 Final Graph-Resolved Answer: {final_answer}")
