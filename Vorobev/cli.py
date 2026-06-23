#!/usr/bin/env python3

import argparse
import json
import os
import sys
import time
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import find_cross_markers_auto, find_single_cross


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross Marker Detector: поиск крестиков на изображениях.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры запуска:
  # Одиночный крестик (одно фото):
  python cli.py --mode SINGLE --input data/photo.jpg --output results/

  # Одиночные крестики (вся папка):
  python cli.py --mode SINGLE --input data/crosses/ --output results/

  # Сетка 9x11 (одно фото):
  python cli.py --mode GRID --input data/grid.jpg --output results/ --grid 9 11

  # Через конфиг JSON:
  python cli.py --config config.json

  # С отображением графиков:
  python cli.py --mode SINGLE --input data/ --output results/ --show

  # Режим Debug (сохраняет промежуточные шаги):
  python cli.py --mode SINGLE --input data/ --output results/ --debug
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Путь к JSON-файлу конфигурации. Перекрывает все остальные аргументы.'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['SINGLE', 'GRID'],
        default='SINGLE',
        help='Режим работы: SINGLE (один крестик) или GRID (сетка меток). Default: SINGLE'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=False,
        help='Путь к файлу или директории с изображениями.'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results',
        help='Директория для сохранения результатов. Default: results/'
    )
    parser.add_argument(
        '--grid',
        type=int,
        nargs=2,
        metavar=('ROWS', 'COLS'),
        default=[7, 7],
        help='Размер сетки для режима GRID. Default: 7 7'
    )
    parser.add_argument(
        '--step-mm',
        type=float,
        default=35.0,
        help='Шаг сетки в миллиметрах (только для GRID). Default: 35.0'
    )
    parser.add_argument(
        '--find-axes',
        action='store_true',
        help='Включить поиск систем координат (только для GRID).'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Показывать графики на экране в процессе работы.'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Режим отладки: сохранять промежуточные шаги (plot-файлы).'
    )

    return parser.parse_args()


def load_config(config_path):
    if not os.path.exists(config_path):
        print(f"[ОШИБКА] Файл конфига не найден: {config_path}")
        sys.exit(1)
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_image_files(input_path):

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    if os.path.isfile(input_path):
        if input_path.lower().endswith(valid_extensions):
            return [input_path]
        else:
            print(f"[ОШИБКА] Файл не является изображением: {input_path}")
            sys.exit(1)

    elif os.path.isdir(input_path):
        files = sorted([
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.lower().endswith(valid_extensions)
        ])
        if not files:
            print(f"[ОШИБКА] В папке '{input_path}' не найдено изображений.")
            sys.exit(1)
        return files

    else:
        print(f"[ОШИБКА] Путь не существует: {input_path}")
        sys.exit(1)


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f} сек"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes} мин {secs:.1f} сек"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours} ч {minutes} мин"


def process_files(image_files, output_dir, mode, grid_shape, step_mm,
                  find_axes, show_process, debug):
    os.makedirs(output_dir, exist_ok=True)

    total = len(image_files)
    success_count = 0
    fail_count = 0
    not_found_count = 0

    times_per_file = []

    print(f"\n{'='*60}")
    print(f"  РЕЖИМ:       {mode}")
    print(f"  Файлов:      {total}")
    print(f"  Результаты:  {output_dir}")
    if mode == 'GRID':
        print(f"  Сетка:       {grid_shape[0]} x {grid_shape[1]}")
        print(f"  Шаг (мм):    {step_mm}")
    print(f"{'='*60}\n")

    for idx, input_path in enumerate(image_files, 1):
        base_name = os.path.splitext(os.path.basename(input_path))[0]

        out_json = os.path.join(output_dir, f"{base_name}_result.json")
        out_img = os.path.join(output_dir, f"{base_name}_result.jpg")
        out_plot = os.path.join(output_dir, f"{base_name}_plot.jpg") if debug else None

        print(f"[{idx}/{total}] {base_name}")

        file_start = time.time()

        try:
            if mode == 'SINGLE':
                result, _ = find_single_cross(
                    image_path=input_path,
                    output_json=out_json,
                    output_img=out_img,
                    output_plot_path=out_plot,
                    show_process=show_process
                )
                if result and result[0].get('found') is False:
                    not_found_count += 1
                    print(f"   [!] Крестик не найден")
                else:
                    success_count += 1

            elif mode == 'GRID':
                result, _ = find_cross_markers_auto(
                    image_path=input_path,
                    grid_shape=tuple(grid_shape),
                    step_mm=step_mm,
                    output_json=out_json,
                    output_img=out_img,
                    output_plot_path=out_plot,
                    find_axes=find_axes,
                    show_process=show_process
                )
                success_count += 1

        except FileNotFoundError as e:
            print(f"   [ОШИБКА] {e}")
            fail_count += 1
        except Exception as e:
            print(f"   [ОШИБКА] Необработанное исключение: {e}")
            if debug:
                import traceback
                traceback.print_exc()
            fail_count += 1

        file_elapsed = time.time() - file_start
        times_per_file.append(file_elapsed)
        print(f"   Время: {format_time(file_elapsed)}")

    return {
        'total': total,
        'success': success_count,
        'not_found': not_found_count,
        'failed': fail_count,
        'times': times_per_file
    }


def print_summary(stats, total_elapsed):
    times = stats['times']
    avg_time = sum(times) / len(times) if times else 0
    min_time = min(times) if times else 0
    max_time = max(times) if times else 0

    print(f"\n{'='*60}")
    print(f"  ИТОГИ ОБРАБОТКИ")
    print(f"{'='*60}")
    print(f"  Всего файлов:        {stats['total']}")
    print(f"  Успешно:             {stats['success']}")
    print(f"  Крестик не найден:   {stats['not_found']}")
    print(f"  Ошибки:              {stats['failed']}")
    print(f"{'='*60}")
    print(f"  Общее время:         {format_time(total_elapsed)}")
    print(f"  Среднее на файл:     {format_time(avg_time)}")
    print(f"  Минимальное:         {format_time(min_time)}")
    print(f"  Максимальное:        {format_time(max_time)}")
    print(f"{'='*60}\n")


def main():
    args = parse_args()

    if args.config:
        config = load_config(args.config)
        mode = config.get('mode', 'SINGLE').upper()
        input_path = config.get('input')
        output_dir = config.get('output', 'results')
        grid_shape = config.get('grid_shape', [7, 7])
        step_mm = config.get('step_mm', 35.0)
        find_axes = config.get('find_axes', False)
        show_process = config.get('show_process', False)
        debug = config.get('debug', False)
    else:
        mode = args.mode
        input_path = args.input
        output_dir = args.output
        grid_shape = args.grid
        step_mm = args.step_mm
        find_axes = args.find_axes
        show_process = args.show
        debug = args.debug

    if not input_path:
        print("[ОШИБКА] Укажите путь к файлу или папке через --input или в config.json")
        sys.exit(1)

    image_files = get_image_files(input_path)

    session_start = time.time()

    stats = process_files(
        image_files=image_files,
        output_dir=output_dir,
        mode=mode,
        grid_shape=grid_shape,
        step_mm=step_mm,
        find_axes=find_axes,
        show_process=show_process,
        debug=debug
    )

    total_elapsed = time.time() - session_start

    print_summary(stats, total_elapsed)

    sys.exit(0 if stats['failed'] == 0 else 1)


if __name__ == "__main__":
    main()