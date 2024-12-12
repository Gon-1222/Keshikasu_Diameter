import cv2
import numpy as np
import matplotlib.pyplot as plt

# 画像を読み込む
def load_image(file_path):
    return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

# 2値化処理
def binarize_image(image, threshold=128):
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary_image

# 等価直径を計算する関数
def calculate_equivalent_diameters(binary_image):
    # ラベリング処理
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image)
    
    # 等価直径を格納するリスト
    equivalent_diameters = []

    for i in range(1, num_labels):  # ラベル0は背景なのでスキップ
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 0:  # 面積が正の値のとき
            diameter = np.sqrt(4 * area / np.pi)*20/400
            equivalent_diameters.append(diameter)

    return equivalent_diameters

# ヒストグラムを描画する関数
def plot_histogram(diameters, bin_size=20):
    plt.hist(diameters, bins=bin_size, color='blue', edgecolor='black', alpha=0.7,range=[0.0,50.0])
    plt.title('Equivalent Diameter Distribution')
    plt.xlabel('Equivalent Diameter (pixels)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# 処理画像を表示する関数
def show_images(original, processed):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Processed Image")
    plt.imshow(processed, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# 処理画像を保存する関数
def save_image(image, file_path):
    cv2.imwrite(file_path, image)

# 画像前処理（ブラー処理）
def preprocess_image(image, blur_kernel=(5, 5)):
    # ブラー処理
    blurred_image = cv2.GaussianBlur(image, blur_kernel, 0)
    return blurred_image

# モルフォロジー処理（2値化後に実行）
def apply_morphology(binary_image, morph_kernel=(7, 7)):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel)
    morphed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    return morphed_image

# メイン関数
def main():
    # 画像ファイルのパス
    file_path = 'Pen.jpg'  # 適宜変更してください
    processed_output_path = 'processed_image.png'  # 処理後の画像の保存先

    # 画像の読み込み
    image = load_image(file_path)

    if image is None:
        print("Error: Could not load the image.")
        return

    # 前処理（ブラー）
    preprocessed_image = preprocess_image(image,blur_kernel=(7,7))

    # 2値化処理
    binary_image = binarize_image(preprocessed_image)

    # モルフォロジー処理
    morphed_image = apply_morphology(binary_image)

    # 等価直径を計算
    diameters = calculate_equivalent_diameters(morphed_image)

    # 処理画像を表示
    show_images(image, morphed_image)

    # 処理画像を保存
    save_image(morphed_image, processed_output_path)
    print(f"Processed image saved to {processed_output_path}")

    # 結果を表示
    print(f"Number of particles: {len(diameters)}")
    print(f"Average of Diameter:{sum(diameters)/len(diameters)}")
    # ヒストグラムをプロット
    plot_histogram(diameters)

if __name__ == "__main__":
    main()