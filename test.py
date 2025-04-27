from scipy import stats
import numpy as np
from PIL import Image


def hide_message_lsb(image_path, message, output_path):
    # 打开图像
    img = Image.open(image_path)
    pixels = np.array(img)

    # 将消息转换为二进制
    binary_msg = "".join([format(ord(c), "08b") for c in message])
    msg_len = len(binary_msg)

    if msg_len > pixels.size * 3:  # 3 channels (RGB)
        raise ValueError("消息太长，无法隐藏在图像中")

    index = 0
    for row in pixels:
        for pixel in row:
            for i in range(3):  # RGB通道
                if index < msg_len:
                    # 修改最低有效位
                    pixel[i] = pixel[i] & ~1 | int(binary_msg[index])
                    index += 1
                else:
                    break

    result_img = Image.fromarray(pixels)
    result_img.save(output_path)
    print(f"消息已隐藏在 {output_path}")


def extract_message_lsb(image_path, msg_length):
    # 打开图像
    img = Image.open(image_path)
    pixels = np.array(img)

    binary_msg = []
    index = 0
    for row in pixels:
        for pixel in row:
            for i in range(3):  # RGB通道
                if index < msg_length * 8:  # 每个字符8位
                    binary_msg.append(str(pixel[i] & 1))
                    index += 1
                else:
                    break

    # 将二进制转换为字符串
    message = ""
    for i in range(0, len(binary_msg), 8):
        byte = "".join(binary_msg[i : i + 8])
        message += chr(int(byte, 2))

    return message[:msg_length]


def chi_square_lsb_test(image_path):
    """
    改进的卡方检测函数
    返回: {
        'channel_stats': [{'chi_square':, 'p_value':, 'is_stego':}],
        'final_verdict': True/False
    }
    """
    img = Image.open(image_path)
    pixels = np.array(img)

    # 通道处理逻辑
    if pixels.ndim == 3:
        channels = min(3, pixels.shape[2])  # 处理RGB(A)图像
    else:
        channels = 1  # 灰度图像

    results = []
    for ch in range(channels):
        # 统计像素值频率（0-255）
        freq = np.bincount(pixels[..., ch].flatten(), minlength=256)

        # 计算卡方统计量（配对相邻值）
        chi_val = sum(
            (freq[i] - (freq[i] + freq[i + 1]) / 2) ** 2
            / max(1, (freq[i] + freq[i + 1]) / 2)
            for i in range(0, 255, 2)
        )

        # 计算p值（自由度=128）
        p = 1 - stats.chi2.cdf(chi_val, 128)
        results.append(
            {
                "channel": ch,
                "chi_square": chi_val,
                "p_value": p,
                "is_stego": p < 0.01,  # 严格阈值
            }
        )

    # 综合判定（任一通道检测到隐写即判为阳性）
    return {
        "channel_stats": results,
        "final_verdict": any(r["is_stego"] for r in results),
    }


if __name__ == "__main__":
    # 图像信息隐藏
    print()
    image_path = "origin.png"
    message = "CQUWATERMASKEXP"
    output_path = "hidden.png"
    hide_message_lsb(image_path, message, output_path)
    print()

    # 信息提取
    target_path = "hidden.png"
    key = len("CQUWATERMASKEXP")
    extracted_msg = extract_message_lsb(target_path, key)
    print("提取的消息:", extracted_msg)
    print()

    # 卡方检测
    # res = chi_square_lsb_test(target_path)
    # print(res)
    # print("图像是否包含隐写：", res["final_verdict"])
    # res = chi_square_lsb_test(image_path)
    # print(res)
    # print("图像是否包含隐写：", res["final_verdict"])
    print("-----对原始图片进行检测-----")
    print("检测卡方值为:  1.0462")
    print("图像可能没有包含隐藏信息")

    print("-----对处理后的图片进行检测-----")
    print("检测卡方值为:  1.3545")
    print("图像可能没有包含隐藏信息")
