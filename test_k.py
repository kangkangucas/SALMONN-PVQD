from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("./llama_path/vicuna-7b-v1.1")
print(tok.convert_tokens_to_ids(["<Speech>", "<SpeechHere>", "</Speech>"]))