import os
import fitz  # PyMuPDF

# Directorios
DATA_DIR = "./pdfs"
OUTPUT_DIR = "./extracted_md"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def pdf_to_markdown(pdf_path):
    """Convierte un PDF a texto estructurado en Markdown."""
    doc = fitz.open(pdf_path)
    md_content = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")

        # Encabezado por página
        md_content.append(f"# Página {page_num}\n\n{text.strip()}\n")

    return "\n\n".join(md_content)

# Procesar todos los PDFs
for file in os.listdir(DATA_DIR):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(DATA_DIR, file)
        md_text = pdf_to_markdown(pdf_path)

        # Guardar con extensión .md
        output_path = os.path.join(OUTPUT_DIR, file.replace(".pdf", ".md"))
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_text)

        print(f"✅ Convertido: {file} → {output_path}")

print("🎉 Conversión completada. Archivos en /extracted_md")
