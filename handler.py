import torch
import runpod
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Configuration
MODEL_NAME = "QCRI/Fanar-1-9B-Instruct"
SYSTEM_PROMPT = "أنت مساعد مفيد وذكي. أجب على السؤال المطلوب فقط وبشكل مباشر. لا تخرج عن سياق السؤال المطروح. كن مختصراً ومفيداً ولا تطل في الحديث دون داعٍ."

# Global variables for model and tokenizer
tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if model is None:
        print(f"Initializing Fanar model: {MODEL_NAME}...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        if not tokenizer.pad_token:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        print("Model loaded and quantized in 4-bit.")

def handler(job):
    job_input = job.get("input", {})
    user_prompt = job_input.get("prompt")
    
    if not user_prompt:
        return {"error": "يرجى تقديم سؤال (Please provide a prompt)."}

    # Maintain the Arabic System Prompt in the message flow
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,  # Strict, concise output
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
    return {"response": response}

load_model()
runpod.serverless.start({"handler": handler})
