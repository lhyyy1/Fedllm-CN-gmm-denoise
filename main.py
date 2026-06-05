#测试

from transformers import AutoTokenizer

tok7 = AutoTokenizer.from_pretrained("./models/llama-2-7b-hf")
tok13 = AutoTokenizer.from_pretrained("/home/cmcc/went/models/Llama-2-13b-hf")

print(len(tok7), len(tok13))
print(tok7.get_vocab() == tok13.get_vocab())