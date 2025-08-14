from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(
    files=["data.txt"],
    vocab_size=50000,
    min_frequency=2,
    special_tokens=["<pad>", "<eos>", "<unk>", "<cls>", "<sep>"]
)

tokenizer.save("my_50k_tokenizer.json")
