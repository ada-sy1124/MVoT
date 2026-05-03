from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/chameleon-7b")

# 查看变色龙第一个图像 Token 叫什么名字，一般是 <image_0> 或者 <image_0000> 之类的
print(tokenizer.encode("<image_0000>"))