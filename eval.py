import os
import json
import torch

from nano_gpt import nanoGPT


model_name = "nanoGPT"
model_dir = f"./models/{model_name}"
model_filename = "state_dict.pt"
model_config_filename = "config.json"
encoding_dict_filename = "encoding_dict.json"
decoding_dict_filename = "decoding_dict.json"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

input_text = "CEASER: "

with open(os.path.join(model_dir, model_config_filename), "r") as f:
    nanoGPT_config = json.load(f)
with open(os.path.join(model_dir, encoding_dict_filename), "r") as f:
    encoding_dict = json.load(f)
with open(os.path.join(model_dir, decoding_dict_filename), "r") as f:
    decoding_dict = json.load(f)

vocab_size = len(encoding_dict)

def encode(s):
    return [encoding_dict[c] for c in s]

def decode(token_seq):
    return "".join([decoding_dict[str(i)] for i in token_seq])

llm = nanoGPT(device=device, vocab_size=vocab_size, **nanoGPT_config)
llm.to(device)
state_dict = torch.load(
    os.path.join(model_dir, model_filename),
    weights_only=True,
    map_location=device
)
llm.load_state_dict(state_dict)
llm.eval()

input_tokens = [encoding_dict[c] for c in input_text]
x = torch.tensor([input_tokens], device=device).reshape(1, -1)

print(f"Starting generation...")
generated_tokens = llm.generate(
    x, max_new_tokens=500
)

generated_text = decode(generated_tokens[0].tolist())
print(f"Text generated: {generated_text}")
