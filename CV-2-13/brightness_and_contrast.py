import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os


def load_image(image_path):
    """Загрузка изображения в цветном формате
        Args:
            image_path (str): Путь к файлу изображения для загрузки

        Returns:
            Загруженное изображение в формате RGB.

        Raises:
            FileNotFoundError - Если файл не существует.
            IsADirectoryError - Если указанный путь является директорией, а не файлом.
            IOError - Если не удалось загрузить изображение.
    """
    img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if not os.path.exists(image_path):
        print(f"Ошибка: Файл '{image_path}' не существует.")
        sys.exit(1)

    if not os.path.isfile(image_path):
        print(f"Ошибка: '{image_path}' является директорией, а не файлом.")
        sys.exit(1)

    if img_color is None:
        print(f"Ошибка: Не удалось загрузить изображение '{image_path}'.")
        sys.exit(1)

    # Конвертируем BGR в RGB для правильного отображения в matplotlib
    img_color_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

    return img_color_rgb


def adjust_brightness_contrast(img, alpha=1.0, beta=0):
    """
    Регулирует яркость и контраст изображения с помощью линейного преобразования.

    Формула преобразования: new_img = alpha * img + beta

    Parameters:
        img - Входное изображение.
        alpha : float - Коэффициент контраста.
        beta : int - Коэффициент яркости.

    Returns:
        Изображение с примененными корректировками яркости и контраста.

    Raises:
        TypeError - Если параметры alpha или beta не являются числами.
        ValueError-Если параметр alpha отрицательный.
    """
    # Проверка корректности параметров
    if not isinstance(alpha, (int, float)):
        raise TypeError("Ошибка: параметр alpha должен быть числом")

    if not isinstance(beta, (int, float)):
        raise TypeError("Ошибка: параметр beta должен быть числом")

    if alpha < 0:
        raise ValueError("Ошибка: параметр alpha (контраст) не может быть отрицательным")

    # Применяем преобразование: new_img = alpha * img + beta
    new_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return new_img


def display_images(original, adjusted, alpha, beta):
    """Отображение исходного и обработанного изображений
    Parameters:
        original - Исходное изображение в формате RGB.
        adjusted - Обработанное изображение с примененными корректировками.
        alpha  - Значение коэффициента контраста, использованное для обработки.
        beta - Значение коэффициента яркости, использованное для обработки.

    Raises:
        RuntimeError - Если произошла ошибка при отображении изображений.
    """
    try:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Отображаем исходное изображение
        axes[0].imshow(original)
        axes[0].set_title('Исходное изображение')

        # Отображаем обработанное изображение
        axes[1].imshow(adjusted)
        axes[1].set_title(f'Контраст: {alpha}, Яркость: {beta}')

        plt.show()
    except Exception as e:
        raise RuntimeError(f"Ошибка при отображении изображений: {str(e)}")


def main():
    """
    Основная функция программы для регулировки яркости и контраста изображения.

    Обрабатывает аргументы командной строки, загружает изображение,
    применяет корректировки и отображает результаты.
    """
    parser = argparse.ArgumentParser(description='Регулировка яркости и контраста изображения')
    parser.add_argument('input_image', help='Путь к исходному изображению')
    parser.add_argument('--alpha', type=float, default=1.5, help='Коэффициент контраста (по умолчанию: 1.5)')
    parser.add_argument('--beta', type=float, default=30, help='Коэффициент яркости (по умолчанию: 30)')
    args = parser.parse_args()

    # Загружаем изображение
    img_color = load_image(args.input_image)

    # Применяем преобразование к цветному изображению
    # Для цветного изображения нужно обработать каждый канал отдельно
    adjusted_color = np.zeros_like(img_color)
    for i in range(3):  # Обрабатываем каждый канал  RGB
        adjusted_color[:, :, i] = adjust_brightness_contrast(img_color[:, :, i],
                                                             alpha=args.alpha,
                                                             beta=args.beta)

    # Показываем цветное изображение
    display_images(img_color, adjusted_color, args.alpha, args.beta)


if __name__ == "__main__":
    main()