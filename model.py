import torch
from tokenizers import Tokenizer
import torch.nn as nn
from torch.nn import functional as F

device = (  # Kodun çalıştığı yerde GPU varsa GPU kullanılır.
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

batch_size = 16  # Tek seferde işlenecek veri havuzu sayısı
block_size = 1024  # Her batch'te işlenecek karakter(token) sayısı

max_iters = 60000  # Toplam eğitim adımı sayısı

eval_iters = 50  # Her değerlendirme sırasında kaç batch kullanılacağı
eval_intervals = 1000  # Kaç adımda bir değerlendirme yapılacağı

n_embd = 384  # Her tokenı temsil eden vektör uzayının boyutu
n_head = 6  # Attention head sayısı
n_layer = 6  # Katman sayısı
dropout = 0.2  # Her eğitim döngüsünde kapatılıcak nöron oranı
learning_rate = 1e-3  # Öğrenme oranı

torch.manual_seed(3040)
# ----------------

with open(
    "/kaggle/input/datasets/kronos1233/last-dance/Gpt_turko_GPT2_cleaned_optimized_v2.txt",
    "r",
    encoding="utf-8",
) as f:
    text = f.read()


chars = sorted(list(set(text)))  # Verideki tüm benzersiz karakterlerin listesi
vocab_size = len(chars)  # Verideki benzersiz karakter sayısı

stoi = {ch: i for i, ch in enumerate(chars)}  # Karakterleri indekse dönüştürür
itos = {i: ch for i, ch in enumerate(chars)}  # İndeksleri karaktere dönüştürür
encode = lambda s: [
    stoi[c] for c in s
]  # Veriyi eğitim için sayılara dönüştüren fonksiyon
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # Inference(çıktı) için sayıları karaktere dönüştüren fonksiyon


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[
    :n
]  # Verinin %90'ı eğitim için kullanılır. Kalan %10'u ise modelin doğruluğunu test etmek için ayrıldı.
eval_data = data[n:]


def get_batch(name):  # Her veri havuzunu(batch) oluşturan fonksiyon
    data = train_data if name == "train" else eval_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():  # Modelin Loss değerlerini hesaplayan fonksiyon
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class GPT_turko(nn.Module):  # Modelin (sadece decoder) ana sınıfı

    def __init__(self):

        super().__init__()
        self.tok_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(  # Modelin katmanlarını oluşturur
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):  # Modelin ağırlıklarını başlatan fonksiyon
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):

        B, T = idx.shape

        tok_embd = self.tok_embedding_table(idx)
        pos_embd = self.pos_embedding_table(torch.arange(T, device=device))
        x = tok_embd + pos_embd
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None

        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):  # Inference için kullanıcağımız fonksiyon
        for _ in range(max_new_tokens):

            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


class Block(nn.Module):  # Her büyük katmanını oluşturan sınıf

    def __init__(self, n_embd, n_head):

        super().__init__()
        head_size = n_embd // n_head
        self.sa_n = Multi_Head_Attention(n_head, head_size)
        self.ff_n = Feed_Forward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):

        x = x + self.sa_n(self.ln1(x))
        x = x + self.ff_n(self.ln2(x))
        return x


class Multi_Head_Attention(nn.Module):  # Tüm attention head'lerini birleştiren sınıf

    def __init__(self, n_head, head_size):

        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))

        return out


class Head(nn.Module):  # Tek bir attention head'ini oluşturan sınıf

    def __init__(self, head_size):

        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) * (self.head_size) ** (-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v

        return out


class Feed_Forward(
    nn.Module
):  # Attention katmanından sonra gelen ve veriyi işleyen katman

    def __init__(self, n_embd):

        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):

        return self.seq(x)


model = GPT_turko()  # model oluşturulur
model = model.to(device)  # model GPU'ya taşınır(varsa)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=learning_rate
)  # optimizer olarak AdamW kullanılır

for iter in range(max_iters):  # Eğitim döngüsü
    if (
        iter % eval_intervals == 0 or iter == max_iters - 1
    ):  # Her eval_intervals adımda veya son adımda Loss hesaplanır ve yazdırılır
        losses = estimate_loss()
        print(
            f"step {iter}: train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f} "
        )

    xb, yb = get_batch("train")  # Eğitim verisi batch'leri alınır

    logits, loss = model(xb, yb)  # Modelin çıktısı ve loss değeri hesaplanır
    optimizer.zero_grad(set_to_none=True)  # Gradientlar sıfırlanır
    loss.backward()  # Backpropagation ile gradientlar hesaplanır
    optimizer.step()  # Hesaplanan gradientlar ile modelin ağırlıkları güncellenir


checkpoint = {  # Modelin ağırlıkları ve optimizer durumu kaydedilir
    "iter": max_iters,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}
torch.save(checkpoint, "gpt_turko_checkpoint.pt")


model.eval()  # model inference moduna alınır.

context = torch.zeros(
    (1, 1), dtype=torch.long, device=device
)  # Modelin başlangıç girdisi olarak tek bir token (0) kullanılır.

with torch.no_grad():  # Modelin ağırlıkları güncellenmeyeceği için gradient hesaplamaları kapatılır.
    generated = model.generate(context, max_new_tokens=1000)  # Çıktı üretilir.

print(
    decode(generated[0].tolist())
)  # Üretilen çıktılar sayılardan karaktere dönüştürülür ve ekrana yazılır.
