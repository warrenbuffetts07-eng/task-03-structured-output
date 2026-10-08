from pathlib import Path
import math
import re
from collections import Counter

ROOT = Path(__file__).parent
DOCS = ROOT / "documents"
TOP_K = 3
MIN_SCORE = 0.12


def chunk_text(text, size=500, overlap=80):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        part = text[start:end].strip()
        if part:
            chunks.append(part)
        if end == len(text):
            break
        start = end - overlap
    return chunks


def load_documents():
    records = []
    for path in sorted(DOCS.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for i, chunk in enumerate(chunk_text(text), 1):
            records.append({
                "source": path.name,
                "chunk": i,
                "text": chunk
            })
    if not records:
        raise RuntimeError("No .txt files found in documents/")
    return records


# Free, local TF-IDF embeddings. No API and no internet are needed.
TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def build_embeddings(records):
    token_lists = [tokenize(r["text"]) for r in records]
    n = len(records)
    document_frequency = Counter()

    for tokens in token_lists:
        document_frequency.update(set(tokens))

    vectors = []
    for tokens in token_lists:
        counts = Counter(tokens)
        total = max(len(tokens), 1)
        vector = {}
        for word, count in counts.items():
            tf = count / total
            idf = math.log((1 + n) / (1 + document_frequency[word])) + 1
            vector[word] = tf * idf
        vectors.append(vector)

    for record, vector in zip(records, vectors):
        record["embedding"] = vector

    return records


def cosine_sparse(a, b):
    dot = sum(value * b.get(word, 0.0) for word, value in a.items())
    na = math.sqrt(sum(value * value for value in a.values()))
    nb = math.sqrt(sum(value * value for value in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def query_embedding(query, records):
    # Use the same TF-IDF vocabulary/IDF as the document index.
    token_lists = [tokenize(r["text"]) for r in records]
    n = len(records)
    document_frequency = Counter()
    for tokens in token_lists:
        document_frequency.update(set(tokens))

    tokens = tokenize(query)
    counts = Counter(tokens)
    total = max(len(tokens), 1)

    vector = {}
    for word, count in counts.items():
        if word in document_frequency:
            tf = count / total
            idf = math.log((1 + n) / (1 + document_frequency[word])) + 1
            vector[word] = tf * idf
    return vector


def retrieve(query, records):
    q_vector = query_embedding(query, records)
    scored = [
        (cosine_sparse(q_vector, r["embedding"]), r)
        for r in records
    ]
    return sorted(scored, key=lambda x: x[0], reverse=True)[:TOP_K]


def answer_from_passages(query, results):
    # A simple extractive answerer. It uses ONLY retrieved document sentences.
    query_words = set(tokenize(query))
    candidates = []

    for score, record in results:
        sentences = re.split(r"(?<=[.!?])\s+", record["text"])
        for sentence in sentences:
            words = set(tokenize(sentence))
            overlap = len(query_words & words)
            if overlap:
                candidates.append((overlap, score, record, sentence.strip()))

    if not candidates:
        return "Not in the documents.", []

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best = candidates[:2]

    # Require at least one meaningful overlap.
    if best[0][0] < 1 or best[0][1] < MIN_SCORE:
        return "Not in the documents.", []

    answer_parts = []
    citations = []
    for _, _, record, sentence in best:
        if sentence and sentence not in answer_parts:
            answer_parts.append(sentence)
            citations.append(f"[{record['source']}, chunk {record['chunk']}]")

    return " ".join(answer_parts), citations


def run_question(query, records):
    results = retrieve(query, records)

    print("\nRetrieved passages:")
    if not results or results[0][0] < MIN_SCORE:
        print("No relevant passage found.")
        print("\nAnswer:\nNot in the documents.")
        return

    for score, record in results:
        print(
            f"\n[{record['source']} | chunk {record['chunk']} | "
            f"similarity {score:.3f}]"
        )
        print(record["text"])

    answer, citations = answer_from_passages(query, results)
    print("\nAnswer:")
    print(answer)
    if citations:
        print("Sources:", " ".join(citations))


if __name__ == "__main__":
    print("Building a FREE local RAG index...")
    records = build_embeddings(load_documents())
    print(f"Indexed {len(records)} document chunks.")
    print("\nNo OpenAI API, API key, or paid service is used.")
    print("Type 'exit' to stop.")

    while True:
        query = input("\nQuestion: ").strip()
        if query.lower() == "exit":
            break
        if query:
            run_question(query, records)
