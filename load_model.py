import torch
from transformers import AutoTokenizer, Gemma3ForCausalLM
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_llm():
    model_name = "google/gemma-3-1b-pt"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = Gemma3ForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        print(f"Model moved to: {model.device}")
    else:
        print("CUDA not available, running on CPU.")

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    return llm
