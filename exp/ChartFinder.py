import fitz  # PyMuPDF
import os

def convert_pdf_to_images_with_fitz(pdf_path, output_folder='output_images_fitz', dpi=200):
    """
    Converts each page of a PDF into images using PyMuPDF (fitz).
    """
    zoom = dpi / 72  # PDF 的預設 DPI 是 72
    mat = fitz.Matrix(zoom, zoom)

    os.makedirs(output_folder, exist_ok=True)

    doc = fitz.open(pdf_path)
    print(f"📄 總頁數：{len(doc)} 頁")

    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat)
        img_path = os.path.join(output_folder, f"page_{i+1}.png")
        pix.save(img_path)
        print(f"✅ 第 {i+1} 頁已轉換 → {img_path}")

    print("🎉 全部頁面轉換完成！")

# 使用方式
pdf_path = './1Q25_MP_Chinese_Vupload.pdf'
convert_pdf_to_images_with_fitz(pdf_path)

