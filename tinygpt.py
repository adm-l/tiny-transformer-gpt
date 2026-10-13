"""A small educational GPT-like decoder-only Transformer. See README for commands."""
import argparse, json, random
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

class CharTokenizer:
    def __init__(self, text):
        chars=sorted(set(text))
        if not chars: raise ValueError("Training text is empty")
        self.stoi={c:i for i,c in enumerate(chars)}; self.itos={i:c for c,i in self.stoi.items()}
    def encode(self, text): return [self.stoi[c] for c in text if c in self.stoi]
    def decode(self, ids): return "".join(self.itos[int(i)] for i in ids if int(i) in self.itos)
    def payload(self): return {"stoi":self.stoi}
    @classmethod
    def load(cls, data):
        obj=cls.__new__(cls); obj.stoi={k:int(v) for k,v in data["stoi"].items()}
        obj.itos={v:k for k,v in obj.stoi.items()}; return obj

class Attention(nn.Module):
    def __init__(self, dim, heads, block, dropout):
        super().__init__(); assert dim%heads==0
        self.heads=heads; self.hdim=dim//heads
        self.qkv=nn.Linear(dim,3*dim); self.proj=nn.Linear(dim,dim)
        self.drop=nn.Dropout(dropout)
        self.register_buffer("mask",torch.tril(torch.ones(block,block))[None,None,:,:],persistent=False)
    def forward(self,x):
        b,t,c=x.shape
        q,k,v=self.qkv(x).chunk(3,dim=-1)
        shape=lambda z:z.view(b,t,self.heads,self.hdim).transpose(1,2)
        q,k,v=map(shape,(q,k,v))
        scores=(q@k.transpose(-2,-1))*(self.hdim**-0.5)
        scores=scores.masked_fill(self.mask[:,:,:t,:t]==0,float("-inf"))
        weights=self.drop(F.softmax(scores,dim=-1))
        y=(weights@v).transpose(1,2).contiguous().view(b,t,c)
        return self.drop(self.proj(y))

class Block(nn.Module):
    def __init__(self,dim,heads,block,dropout):
        super().__init__(); self.n1=nn.LayerNorm(dim); self.attn=Attention(dim,heads,block,dropout)
        self.n2=nn.LayerNorm(dim); self.ff=nn.Sequential(nn.Linear(dim,4*dim),nn.GELU(),nn.Linear(4*dim,dim),nn.Dropout(dropout))
    def forward(self,x):
        x=x+self.attn(self.n1(x)); return x+self.ff(self.n2(x))

class TinyGPT(nn.Module):
    def __init__(self,vocab_size,block_size=128,n_embd=128,n_head=4,n_layer=4,dropout=.1):
        super().__init__()
        if n_embd%n_head: raise ValueError("n_embd must be divisible by n_head")
        self.cfg=dict(vocab_size=vocab_size,block_size=block_size,n_embd=n_embd,n_head=n_head,n_layer=n_layer,dropout=dropout)
        self.tok=nn.Embedding(vocab_size,n_embd); self.pos=nn.Embedding(block_size,n_embd); self.drop=nn.Dropout(dropout)
        self.blocks=nn.Sequential(*[Block(n_embd,n_head,block_size,dropout) for _ in range(n_layer)])
        self.norm=nn.LayerNorm(n_embd); self.head=nn.Linear(n_embd,vocab_size,bias=False); self.head.weight=self.tok.weight
        self.apply(self.init_weights)
    @staticmethod
    def init_weights(m):
        if isinstance(m,(nn.Linear,nn.Embedding)):
            nn.init.normal_(m.weight,mean=0,std=.02)
            if isinstance(m,nn.Linear) and m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self,idx,targets=None):
        b,t=idx.shape
        if t>self.cfg["block_size"]: raise ValueError("Sequence longer than block_size")
        p=torch.arange(t,device=idx.device); x=self.drop(self.tok(idx)+self.pos(p))
        logits=self.head(self.norm(self.blocks(x)))
        loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1)) if targets is not None else None
        return logits,loss
    @torch.no_grad()
    def generate(self,idx,tokens=100,temperature=.8,top_k=20):
        if temperature<=0: raise ValueError("temperature must be positive")
        self.eval()
        for _ in range(tokens):
            logits,_=self(idx[:,-self.cfg["block_size"]:]); logits=logits[:,-1,:]/temperature
            if top_k:
                k=min(max(1,top_k),logits.size(-1)); vals,_=torch.topk(logits,k); logits[logits<vals[:,-1,None]]=float("-inf")
            idx=torch.cat((idx,torch.multinomial(F.softmax(logits,dim=-1),1)),dim=1)
        return idx

def main():
    p=argparse.ArgumentParser(); p.add_argument("command",choices=["train","generate"]); p.add_argument("--data",default="data.txt")
    p.add_argument("--checkpoint",default="checkpoint.pt"); p.add_argument("--steps",type=int,default=1000); p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--block-size",type=int,default=128); p.add_argument("--n-embd",type=int,default=128); p.add_argument("--n-head",type=int,default=4)
    p.add_argument("--n-layer",type=int,default=4); p.add_argument("--lr",type=float,default=3e-4); p.add_argument("--prompt",default="Once upon a time")
    p.add_argument("--tokens",type=int,default=200); p.add_argument("--temperature",type=float,default=.8); a=p.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"; random.seed(1337); torch.manual_seed(1337)
    if a.command=="train":
        text=Path(a.data).read_text(encoding="utf-8"); tok=CharTokenizer(text); data=torch.tensor(tok.encode(text),dtype=torch.long)
        if len(data)<=a.block_size+1: raise ValueError("Training corpus must exceed block_size+1 characters; lower --block-size or add data")
        cfg=dict(vocab_size=len(tok.stoi),block_size=a.block_size,n_embd=a.n_embd,n_head=a.n_head,n_layer=a.n_layer)
        model=TinyGPT(**cfg).to(device); opt=torch.optim.AdamW(model.parameters(),lr=a.lr)
        for step in range(1,a.steps+1):
            starts=torch.randint(len(data)-a.block_size-1,(a.batch_size,))
            x=torch.stack([data[i:i+a.block_size] for i in starts]).to(device)
            y=torch.stack([data[i+1:i+a.block_size+1] for i in starts]).to(device)
            _,loss=model(x,y); opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            if step==1 or step%100==0 or step==a.steps: print(f"step={step} loss={loss.item():.4f} device={device}")
        Path(a.checkpoint).parent.mkdir(parents=True,exist_ok=True)
        torch.save({"config":model.cfg,"tokenizer":tok.payload(),"state_dict":model.cpu().state_dict()},a.checkpoint)
        print("Saved",a.checkpoint)
    else:
        ck=torch.load(a.checkpoint,map_location=device,weights_only=True); tok=CharTokenizer.load(ck["tokenizer"])
        model=TinyGPT(**ck["config"]).to(device); model.load_state_dict(ck["state_dict"]); ids=tok.encode(a.prompt) or [0]
        result=model.generate(torch.tensor([ids],device=device),a.tokens,a.temperature)
        print(tok.decode(result[0].tolist()))
if __name__=="__main__": main()
