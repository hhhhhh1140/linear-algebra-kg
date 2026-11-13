"""模型：CRF、BiLSTM-CRF（命名实体识别）、BiLSTM 关系分类器，以及字符/BERT 两种输入表示。"""
from __future__ import annotations

import torch
import torch.nn as nn

from common import ENTITY_TYPES, NER_LABELS, REL_LABELS, OUT

PAD, UNK = "<pad>", "<unk>"
TYPE_LIST = ["NONE"] + list(ENTITY_TYPES)


# ------------------------------------------------------------------ 输入表示
class CharVocab:
    def __init__(self, texts):
        chars = sorted({c for t in texts for c in t})
        self.itos = [PAD, UNK] + chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def encode(self, text):
        return [self.stoi.get(c, 1) for c in text]


class BertFeaturizer:
    """冻结的 bert-base-chinese：为每个汉字取最后一层隐向量（首个子词），结果缓存到磁盘。"""

    NAME = "bert-base-chinese"

    def __init__(self, device=None):
        from transformers import AutoModel, AutoTokenizer
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(self.NAME)
        self.model = AutoModel.from_pretrained(self.NAME).to(self.device).eval()
        self.cache_path = OUT / "bert_cache.pt"
        self.cache = torch.load(self.cache_path) if self.cache_path.exists() else {}

    @torch.no_grad()
    def featurize(self, texts, bs=32):
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        for i in range(0, len(todo), bs):
            batch = todo[i:i + bs]
            enc = self.tok([list(t) for t in batch], is_split_into_words=True, return_tensors="pt",
                           padding=True, truncation=True, max_length=256)
            hid = self.model(**{k: v.to(self.device) for k, v in enc.items()}).last_hidden_state.cpu()
            for b, t in enumerate(batch):
                feat = torch.zeros(len(t), hid.size(-1))
                seen = set()
                for pos, w in enumerate(enc.word_ids(b)):
                    if w is not None and w not in seen:
                        seen.add(w)
                        feat[w] = hid[b, pos]
                self.cache[t] = feat.half()
        if todo:
            OUT.mkdir(exist_ok=True)
            torch.save(self.cache, self.cache_path)
        return [self.cache[t].float() for t in texts]


class Encoder(nn.Module):
    """把一批句子编码成逐字向量：char 模式用可训练字向量，bert 模式用冻结 BERT 特征。"""

    def __init__(self, mode, vocab_size=0, emb_dim=128, bert_dim=768):
        super().__init__()
        self.mode = mode
        if mode == "char":
            self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            self.dim = emb_dim
        else:
            self.proj = nn.Sequential(nn.Dropout(0.2), nn.Linear(bert_dim, 256), nn.GELU())
            self.dim = 256

    def forward(self, x):
        return self.emb(x) if self.mode == "char" else self.proj(x)


# ------------------------------------------------------------------ CRF
class CRF(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.trans = nn.Parameter(torch.zeros(n, n))   # trans[i, j]: 标签 i -> j
        self.start = nn.Parameter(torch.zeros(n))
        self.end = nn.Parameter(torch.zeros(n))

    def nll(self, em, tags, mask):
        B, T, n = em.shape
        idx = torch.arange(B)
        gold = self.start[tags[:, 0]] + em[idx, 0, tags[:, 0]]
        for t in range(1, T):
            step = self.trans[tags[:, t - 1], tags[:, t]] + em[idx, t, tags[:, t]]
            gold = gold + step * mask[:, t]
        last = tags[idx, mask.sum(1) - 1]
        gold = gold + self.end[last]

        alpha = self.start + em[:, 0]
        for t in range(1, T):
            nxt = torch.logsumexp(alpha.unsqueeze(2) + self.trans + em[:, t].unsqueeze(1), dim=1)
            alpha = torch.where(mask[:, t, None].bool(), nxt, alpha)
        logz = torch.logsumexp(alpha + self.end, dim=1)
        return (logz - gold).mean()

    @torch.no_grad()
    def decode(self, em, mask):
        B, T, n = em.shape
        score = self.start + em[:, 0]
        back = []
        for t in range(1, T):
            s = score.unsqueeze(2) + self.trans
            best, arg = s.max(1)
            nxt = best + em[:, t]
            m = mask[:, t, None].bool()
            score = torch.where(m, nxt, score)
            back.append(torch.where(m, arg, torch.arange(n).expand(B, n)))
        score = score + self.end
        paths = []
        lengths = mask.sum(1).tolist()
        for b in range(B):
            tag = int(score[b].argmax())
            path = [tag]
            for t in range(T - 2, -1, -1):
                tag = int(back[t][b, tag])
                path.append(tag)
            path.reverse()
            paths.append(path[: lengths[b]])
        return paths


# ------------------------------------------------------------------ NER
class BiLSTMCRF(nn.Module):
    def __init__(self, mode, vocab_size=0, hidden=128):
        super().__init__()
        self.enc = Encoder(mode, vocab_size)
        self.drop = nn.Dropout(0.3)
        self.lstm = nn.LSTM(self.enc.dim, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hidden, len(NER_LABELS))
        self.crf = CRF(len(NER_LABELS))

    def emissions(self, x, mask):
        h = self.drop(self.enc(x))
        packed = nn.utils.rnn.pack_padded_sequence(h, mask.sum(1).cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        return self.fc(self.drop(out))

    def loss(self, x, tags, mask):
        return self.crf.nll(self.emissions(x, mask), tags, mask)

    def predict(self, x, mask):
        return self.crf.decode(self.emissions(x, mask), mask)


# ------------------------------------------------------------------ 关系抽取
class BiLSTMRelation(nn.Module):
    """输入：句子逐字表示 + 到头/尾实体的相对位置 + 头/尾实体标记；
    输出：头实体 -> 尾实体 的关系类别（含 NA）。"""

    MAXD = 40

    def __init__(self, mode, vocab_size=0, hidden=128):
        super().__init__()
        self.enc = Encoder(mode, vocab_size)
        self.pos_h = nn.Embedding(2 * self.MAXD + 1, 24)
        self.pos_t = nn.Embedding(2 * self.MAXD + 1, 24)
        self.mark = nn.Embedding(3, 16)          # 0 其他, 1 头实体, 2 尾实体
        self.type_emb = nn.Embedding(len(TYPE_LIST), 16)
        self.drop = nn.Dropout(0.3)
        self.lstm = nn.LSTM(self.enc.dim + 24 + 24 + 16, hidden, batch_first=True, bidirectional=True)
        self.cls = nn.Sequential(nn.Linear(6 * hidden + 32, 256), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(256, len(REL_LABELS)))

    def forward(self, x, mask, ph, pt, mark, htype, ttype):
        h = torch.cat([self.enc(x), self.pos_h(ph), self.pos_t(pt), self.mark(mark)], -1)
        h = self.drop(h)
        packed = nn.utils.rnn.pack_padded_sequence(h, mask.sum(1).cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        neg = torch.finfo(out.dtype).min
        pooled = out.masked_fill(~mask.bool().unsqueeze(-1), neg).max(1).values
        hm = (mark == 1).float().unsqueeze(-1)
        tm = (mark == 2).float().unsqueeze(-1)
        head = (out * hm).sum(1) / hm.sum(1).clamp(min=1)
        tail = (out * tm).sum(1) / tm.sum(1).clamp(min=1)
        z = torch.cat([pooled, head, tail, self.type_emb(htype), self.type_emb(ttype)], -1)
        return self.cls(z)
