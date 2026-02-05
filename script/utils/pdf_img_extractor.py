import fitz  # PyMuPDF
import os
from pdf2image import convert_from_path

def extract_images_from_pdf(pdf_path, output_folder="extracted_images"):
    # 如果输出文件夹不存在，则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开 PDF 文件
    pdf_file = fitz.open(pdf_path)
    
    image_count = 0

    # 遍历每一页
    for page_index in range(len(pdf_file)):
        page = pdf_file[page_index]
        image_list = page.get_images(full=True)

        # 遍历当前页的所有图片
        for img_index, img in enumerate(image_list):
            xref = img[0]  # 图片的 XREF 编号
            base_image = pdf_file.extract_image(xref)
            image_bytes = base_image["image"]  # 图片字节数据
            image_ext = base_image["ext"]    # 图片扩展名 (如 png, jpeg)
            
            # 命名规则：第几页_第几个图片
            image_name = f"page{page_index+1}_img{img_index+1}.{image_ext}"
            image_path = os.path.join(output_folder, image_name)

            # 保存图片
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            
            print(f"已保存: {image_name}")
            image_count += 1

    pdf_file.close()
    print(f"\n提取完成！总共提取了 {image_count} 张图片。")


## TODO 有问题
def export_pdf_to_high_res_images(pdf_path):
    # 建议设置 dpi=300
    images = convert_from_path(pdf_path, dpi=600)

    for i, image in enumerate(images):
        # 保存为 PNG 格式通常比 JPEG 更清晰，因为 PNG 是无损压缩
        image.save(f'high_res_page_{i+1}.png', 'PNG')
        print(f"高清图片第 {i+1} 页已生成")

if __name__ == "__main__":
    # 使用示例
    # extract_images_from_pdf("d:/0-mingyang/img_handle/场馆展位图/中药展展位图（8.22V3）.pdf")
    export_pdf_to_high_res_images("d:/0-mingyang/img_handle/场馆展位图/中药展展位图（8.22V3）.pdf")