# Task 03 — Free Local Retrieval-Augmented Answers

This project demonstrates Retrieval-Augmented Generation (RAG) without using a paid API.

## What it demonstrates

- Documents are split into chunks.
- Each chunk is converted into a local TF-IDF vector (a free local embedding-style representation).
- The most relevant passages are retrieved for a question.
- Retrieved passages are displayed before the answer.
- The answer is extracted only from the retrieved documents.
- If the information is not available, the program honestly responds:
  `Not in the documents.`
- Sources are cited using the filename and chunk number.

## Cost

**₹0 API cost.**

This version does NOT use OpenAI, an API key, internet access, or a paid AI service.

## Requirements

- Python 3.10 or newer
- No external Python packages are required.

## How to run

Open PowerShell in this project folder and run:

```powershell
python main.py
```

Then try:

```text
What payment methods are supported?
```

and:

```text
What is the trolley rate?
```

Also test something that is not in the documents:

```text
Who is the owner of VISHAL TRADERS?
```

The expected behavior for unsupported information is:

```text
Not in the documents.
```

## Project structure

```text
task-03-rag-free/
├── main.py
├── README.md
├── .gitignore
└── documents/
    ├── shop_profile.txt
    └── products.txt
```

## How the RAG pipeline works

1. Read `.txt` documents.
2. Split them into overlapping chunks.
3. Create local TF-IDF vector representations.
4. Convert the question into the same representation.
5. Calculate cosine similarity.
6. Retrieve the top relevant passages.
7. Display the retrieved passages.
8. Extract an answer only from those passages.
9. Show the source filename and chunk number.

This keeps the demonstration completely local and free.

## Replacing the sample documents

You can add your own `.txt` documents inside the `documents` folder. Run the program again after changing the documents so the index is rebuilt.
