<div align="center">

# GPT-1 Fine-tuned Model

</div>

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 15px; color: white; margin: 30px 0; box-shadow: 0 8px 32px rgba(0,0,0,0.1);">

## Quick Start

<div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; margin: 20px 0;">

### Live Notebook Available 

- **[Try it on Google Colab](https://colab.research.google.com/drive/1E46Cxuv1t-DuYMuZeB0BnQ4IjMIzaft4?usp=sharing)**

- **[Jump to Example Generations](#example-1-general-qa)**

*Click the link above to test the model's generation capabilities interactively*

**What you'll find:**
- Live inference
- Adjustable hyperparameters  
- Multiple use cases (Q&A, creative writing, code generation)


</div>

</div>

---

## Model Architecture & Training Details

This GPT-1 model is a decoder-only transformer trained from scratch on a single NVIDIA A40 GPU (48GB VRAM). The model was trained in two stages: pretraining on OpenWebText followed by instruction fine-tuning on OpenAssistant OASST2.

### Model Architecture

- **Architecture**: Decoder-only transformer with 12 layers
- **Model dimension**: 1024
- **Attention heads**: 16 (64-dim per head)
- **MLP ratio**: 4x (4096 hidden dimension)
- **Max sequence length**: 512 tokens
- **Vocabulary size**: 51,682 tokens (custom BPE tokenizer)
- **Total parameters**: ~250M (base model)
- **Special tokens**: `[PAD]` (0), `[UNK]` (1), `[EOS]` (2)

### Training Configuration

#### Pretraining (OpenWebText)
- **Dataset**: [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) (~8.8B tokens)
- **Duration**: ~40 hours (1 epoch)
- **Batch size**: 48 sequences per batch
- **Gradient accumulation**: 4 steps (effective batch size: 192)
- **Optimizer**: AdamW (β₁=0.9, β₂=0.95, weight_decay=0.0)
- **Learning rate**: Cosine decay (3e-4 → 1e-6) with 3% warmup
- **Precision**: Mixed precision (FP16 compute, FP32 master weights)
- **Dropout**: 0.0 (no dropout used)

#### Fine-tuning (LoRA on OASST2)
- **Dataset**: [OpenAssistant OASST2](https://huggingface.co/datasets/OpenAssistant/oasst2) (~50M tokens)
- **Duration**: ~70 minutes (1 epoch)
- **Method**: LoRA
  - Rank (r): 8, Alpha: 8
  - Trainable parameters: ~15M
  - Applied to: Q, K, V, O projections + MLP up/down projections
- **Frozen weights**: All base parameters except special token embeddings
- **Loss masking**: Only compute loss on assistant responses
- **Batch size**: 64 sequences
- **Learning rate**: 1e-4 → 1e-6 with cosine decay

### Training Results

- **Final pretraining validation perplexity**: 18.17
- **Final fine-tuned validation perplexity**: 8.05

### Intended Use & Limitations
**Intended Use Cases:**
- General question answering and conversational interactions
- Educational content generation and explanations
- Creative writing assistance

**Limitations:**
- **Parameter count**: As a 250M parameter model, it has horrible specific factual recall
- **Specialized tasks**: Not optimized for domain-specific tasks like medical advice, legal counsel, or financial planning
- **Bias and safety**: Has not been fine-tuned for safety
- **Context length**: Limited to 512 tokens maximum context, restricting long-form conversations or document analysis

---

## Example 1: General Q&A

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; margin: 20px 0;">

### Prompt:
```text
How can I create a balanced diet?
```

</div>

<div style="background: #f8f9fa; padding: 15px; border-left: 4px solid #007bff; border-radius: 5px; margin: 10px 0;">

### Hyperparameters:
- `temperature`: 0.7
- `top_k`: 50
- `repeat_penalty`: None

</div>

<div style="background: #e8f5e8; padding: 15px; border-left: 4px solid #28a745; border-radius: 5px; margin: 10px 0;">

### Model Output:
```text
The diet can be structured in several ways: 
    1. Balance: Start with a healthy balance of fats and proteins, such as saturated fat, carbohydrates, and carbohydrates. 
    2. Dietary Supplements: Regular dietary supplements, such as fruit, vegetables, whole grains, and nuts, are important for a healthy diet. 
    3. Dietary counseling: Regular counseling is important for both parents and children to help ensure they are taking proper care of themselves. 
    4. Food counseling: Regular counseling is an essential part of any healthy eating plan. 5. Nutrition counseling: Eating meals that do not include all foods at once is important to prevent overeating and prevent overeating. 

It's important to note that this is just a starting point. Different diets can have different effects, and some people may still gain weight or gain muscle mass. 

However, it's important to be aware of your own body and to plan ahead!
```

</div>

---

## Example 2: Creative Writing

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; margin: 20px 0;">

### Prompt 1:
```text
Write me a love story.
```

</div>

<div style="background: #f8f9fa; padding: 15px; border-left: 4px solid #007bff; border-radius: 5px; margin: 10px 0;">

### Hyperparameters:
- `temperature`: 0.5
- `top_k`: 40
- `repeat_penalty`: 1.05

</div>

<div style="background: #e8f5e8; padding: 15px; border-left: 4px solid #28a745; border-radius: 5px; margin: 10px 0;">

### Model Output:
```text
I was in the middle of my life when I met my future self. He had been diagnosed with cancer and he knew that there were many things to be grateful for, but one thing stood out most: his new friend.

He had always wanted to be my friend so much that it seemed like every time we talked about something important or interesting, he would say "thank you." He never really cared if anything happened because he just felt so good knowing that everyone else around him wasn't too bad at what they did together!

So after being through this whole process all over again, I decided to give up on him completely. I didn't want to let him down any more than ever before giving up entirely.

And then, as I thought about it now, I realized how much I needed him - even though I know I wouldn't have done that without him. I could not help myself from thinking that everything might go wrong during this entire journey.

Now, I'm glad to hear that I finally gave up hope...but still, I couldn't resist taking solace in the fact that I could do something truly special once again.
```

</div>

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; margin: 20px 0;">

### Prompt 2:
```text
Write me a short story about alien abduction.
```

</div>

<div style="background: #f8f9fa; padding: 15px; border-left: 4px solid #007bff; border-radius: 5px; margin: 10px 0;">

### Hyperparameters:
- `temperature`: 0.5
- `top_k`: 40
- `repeat_penalty`: 1.05

</div>

<div style="background: #e8f5e8; padding: 15px; border-left: 4px solid #28a745; border-radius: 5px; margin: 10px 0;">

### Model Output:
```text
As I was reading this book, my heart started to race around the room as I listened intently for clues about what was going on in the world that day.

I felt like I had been transported into a different dimension of reality by an ancient alien who lived tens of thousands of years ago and gave us all his wisdom!

My brain went through some really weird things when I began to read books again... but it wasn't until after I'd finished reading the first chapter that I realized how much more bizarrely fantastical my thoughts were than they should be at any moment.

And then I remembered something: my mind is so fucking crazy because of these strange creatures living inside our bodies....and yet we're still mesmerized with them now!!!!!

So yeah, I loved reading books back then -- especially those that told stories about aliens or supernatural beings from the past!

But alas, I have no idea why I've not become one since then. I just know that there's nothing quite like it anymore without knowing the answers too :)
```

</div>

---

## Example 3: Code Generation

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; margin: 20px 0;">

### Prompt:
```text
Write me a python function to print "Hello, world!".
```

</div>

<div style="background: #f8f9fa; padding: 15px; border-left: 4px solid #007bff; border-radius: 5px; margin: 10px 0;">

### ⚙️ Hyperparameters:
- `temperature`: 0.2
- `top_k`: 30
- `top_p`: 0.8
- `repeat_penalty`: 1.05

</div>

<div style="background: #e8f5e8; padding: 15px; border-left: 4px solid #28a745; border-radius: 5px; margin: 10px 0;">

### Model Output:
``````
```python
print("Hello, world!")
```

This will print the message "Hello, world!".

If you want to use this function in your web page or any other application that uses Python, please let me know.
``````

</div>

