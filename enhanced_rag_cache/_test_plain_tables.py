import sys
sys.path.insert(0, r'C:\Users\z036635\Desktop\lang_proj\rag_architect\enhanced_rag_cache')
from src.utils.pdf_to_markdown import file_to_text
from src.chunking.structure_recursive import extract_special_blocks, structure_recursive_chunk
from pathlib import Path

PDF_PATH = r'C:\Users\z036635\Desktop\lang_proj\rag_architect\enhanced_rag_cache\docs\2602.03442v1_rag_pdf-1-9.pdf'

md = file_to_text(PDF_PATH, embed_images=False)

print("=== extract_special_blocks (pymupdf4llm table_strategy=text) ===")
cleaned, blocks = extract_special_blocks(md)
print(f"Total blocks: {len(blocks)}")
for i, b in enumerate(blocks):
    preview = b["content"][:90].replace("\n", " ")
    print(f"  [{i}] type={b['type']:6}  rows={b.get('table_rows','n/a')}  cols={b.get('table_columns','n/a')}  {preview}")

print()
print("=== structure_recursive_chunk (pymupdf4llm table_strategy=text) ===")
chunks = structure_recursive_chunk(md, doc_id="test.pdf", source="test.pdf")
n_text  = sum(1 for c in chunks if c.chunk_type == "text")
n_table = sum(1 for c in chunks if c.chunk_type == "table")
n_image = sum(1 for c in chunks if c.chunk_type == "image")
print(f"Total chunks: {len(chunks)}  text={n_text}  table={n_table}  image={n_image}")
print()
print("--- table chunks ---")
for c in chunks:
    if c.chunk_type == "table":
        preview = c.text[:120].replace("\n", " ")
        print(f"  {c.chunk_id}  heading={c.heading_meta}  meta={c.metadata}")
        print(f"    preview: {preview}")
