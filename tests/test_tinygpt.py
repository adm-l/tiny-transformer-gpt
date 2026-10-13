import torch
from tinygpt import CharTokenizer,TinyGPT

def test_tokenizer_roundtrip():
    t=CharTokenizer("hello world")
    assert t.decode(t.encode("hello"))=="hello"

def test_forward_loss_and_shape():
    m=TinyGPT(16,block_size=8,n_embd=16,n_head=4,n_layer=2,dropout=0)
    x=torch.randint(0,16,(2,8)); logits,loss=m(x,x)
    assert logits.shape==(2,8,16) and torch.isfinite(loss)

def test_generation_length():
    m=TinyGPT(8,block_size=8,n_embd=16,n_head=4,n_layer=1,dropout=0)
    assert m.generate(torch.zeros((1,3),dtype=torch.long),tokens=2).shape==(1,5)
