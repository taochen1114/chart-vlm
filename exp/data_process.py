import argparse

# import fitz  # PyMuPDF
import pymupdf as fitz
import os

def convert_pdf_to_images_with_fitz(pdf_path="", output_folder='output_images_fitz', dpi=200):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_path", type=str, required=True, default='./data/pdf/1Q25_MP_Chinese_Vupload.pdf')
    parser.add_argument("--output_folder", type=str, default='')
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()    


    pdf_fname = os.path.basename(args.pdf_path)

    print(f"pdf_fname {pdf_fname}")

    if args.output_folder == '':
        args.output_folder = os.path.join('./data/pdf_img', pdf_fname.split('.')[0])

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder, exist_ok=True)
        print(f"📁 創建輸出資料夾：{args.output_folder}")

    convert_pdf_to_images_with_fitz(args.pdf_path, args.output_folder)


