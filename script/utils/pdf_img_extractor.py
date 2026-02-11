from pathlib import Path

import fitz  # PyMuPDF
from pdf2image import convert_from_path

# 导入工程化工具
from src.utils import get_logger, safe_mkdir

logger = get_logger('utils.pdf_img_extractor')


def extract_images_from_pdf(pdf_path, output_folder="extracted_images"):
    """
    从PDF中提取嵌入的图像

    参数:
        pdf_path: PDF文件路径
        output_folder: 输出文件夹路径
    """
    safe_mkdir(output_folder)

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
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            # 命名规则：第几页_第几个图片
            image_name = f"page{page_index+1}_img{img_index+1}.{image_ext}"
            image_path = Path(output_folder) / image_name

            # 保存图片
            with open(image_path, "wb") as f:
                f.write(image_bytes)

            logger.info(f"已保存: {image_name}")
            image_count += 1

    pdf_file.close()
    logger.info(f"提取完成！总共提取了 {image_count} 张图片。")


def export_pdf_to_high_res_images(pdf_path, output_folder="high_res_images", dpi=600):
    """
    将PDF的每一页转换为高分辨率图像

    参数:
        pdf_path: PDF文件路径
        output_folder: 输出文件夹路径
        dpi: 分辨率（DPI），默认600
    """
    safe_mkdir(output_folder)
    logger.info(f"开始将PDF转换为高清图像，DPI: {dpi}")

    images = convert_from_path(pdf_path, dpi=dpi)

    for i, image in enumerate(images):
        # 保存为 PNG 格式通常比 JPEG 更清晰，因为 PNG 是无损压缩
        image_path = Path(output_folder) / f'high_res_page_{i+1}.png'
        image.save(str(image_path), 'PNG')
        logger.info(f"高清图片第 {i+1} 页已生成")

    logger.info(f"PDF转图像完成！共生成 {len(images)} 页。")


if __name__ == "__main__":
    from src.utils import get_project_root

    project_root = get_project_root()

    # 使用示例（请根据实际路径修改）
    pdf_path = project_root / 'documents' / 'sample.pdf'

    # extract_images_from_pdf(pdf_path, output_folder=project_root / 'output' / 'extracted_images')
    export_pdf_to_high_res_images(pdf_path, output_folder=project_root / 'output' / 'high_res_images', dpi=600)
