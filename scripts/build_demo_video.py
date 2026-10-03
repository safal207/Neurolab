"""Render a silent 60-second explainer from a trusted lab bundle (requires ffmpeg)."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neurolab.lab_demo import DemoPredictor, EXAMPLES


BACKGROUND = "#0d2031"
WHITE = "#f5f9fd"
MUTED = "#b8cad9"
GOLD = "#d8ac66"


def base(number, title):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BACKGROUND)
    fig.text(.055, .92, "NEUROLAB / ВОСПРОИЗВОДИМЫЙ ML", color=GOLD, fontsize=14, weight="bold")
    fig.text(.055, .83, title, color=WHITE, fontsize=29, weight="bold")
    fig.text(.055, .06, "github.com/safal207/Neurolab · исследовательский кейс на английских текстах", color=MUTED, fontsize=11)
    fig.text(.055, .025, "EmoBank — S. Buechel & U. Hahn (2017), CC-BY-SA 4.0 · код Neurolab: MIT", color=MUTED, fontsize=10)
    fig.text(.94, .055, f"{number}/5", color=MUTED, fontsize=12, ha="right")
    return fig


def lines(fig, content):
    for y, line in zip((.64, .48, .32), content):
        fig.text(.07, y, line, color=WHITE, fontsize=24, linespacing=1.5)


def build(bundle, output):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = output.parent / "neurolab-video-frames"
    frames.mkdir(exist_ok=True)
    report = json.loads((Path(bundle) / "report.json").read_text())
    predictor = DemoPredictor(bundle)
    figures = []

    fig = base(1, "Как проверить пользу модели?")
    lines(fig, ["Обучим модель на английских текстах.",
                "Сравним её с простым методом на новых примерах.",
                "Сохраним веса, результаты и происхождение данных."])
    figures.append(fig)

    fig = base(2, "Проверяем путь от данных до результата")
    lines(fig, ["Train обучает. Dev выбирает настройки.",
                "Test проверяет качество на отложенных текстах.",
                "Правильные ответы входят в расчёт ошибки.\nМодель не получает их на вход."])
    figures.append(fig)

    fig = base(3, "Ошибка на новых текстах · меньше лучше")
    rows = report["test_metrics"]
    names = [row["model"] for row in rows]
    values = [row["MAE_mean"] for row in rows]
    ax = fig.add_axes([.12, .28, .79, .43], facecolor=BACKGROUND)
    bars = ax.bar(names, values, color=["#879cae", "#507d9a", "#559ed0", GOLD], width=.57)
    ax.set_ylim(0, max(values) * 1.28)
    ax.set_ylabel("Средняя MAE · пункты шкалы 1–5", color=WHITE, fontsize=13)
    ax.tick_params(colors=WHITE, labelsize=15)
    ax.spines[["top", "right"]].set_visible(False)
    for name in ("bottom", "left"):
        ax.spines[name].set_color(MUTED)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, value+.01, f"{value:.4f}", ha="center", color=WHITE, fontsize=19)
    comparison = report["paired_comparisons_to_ridge"].get("Neurolab K=5")
    verdict = "Сравнение относится к этому эксперименту и этим отложенным текстам."
    if comparison:
        lower, upper = comparison["row_bootstrap_95pct"]
        if upper < 0:
            verdict = "Ошибка K=5 ниже Ridge в этом запуске: 95% интервал по строкам ниже нуля."
        elif lower > 0:
            verdict = "Ошибка K=5 выше Ridge в этом запуске: 95% интервал по строкам выше нуля."
        else:
            verdict = "Преимущество K=5 над Ridge не установлено в этом эксперименте."
    fig.text(.075, .17, verdict, color=WHITE, fontsize=18)
    counts = report["data"]["used_counts"]
    fig.text(.075, .12, f"Seed {report['config']['seed']}; {counts['train']:,} train / {counts['dev']:,} dev / {counts['test']:,} test.", color=MUTED, fontsize=14)
    figures.append(fig)

    fig = base(4, "Один текст · три сохранённые модели")
    sentence = EXAMPLES["Хорошая новость"]
    table = predictor.predict([sentence])
    fig.text(.065, .7, "\n".join(textwrap.wrap(sentence, 85)), color=WHITE, fontsize=19)
    ax = fig.add_axes([.07, .32, .86, .29])
    ax.axis("off")
    content = [[row["method"], *[f"{row[axis]:.3f}" for axis in ("V", "A", "D")]]
               for _, row in table.iterrows()]
    tab = ax.table(cellText=content,
                   colLabels=["Метод", "V · положительность", "A · активность", "D · контроль"],
                   cellLoc="center", colWidths=[.28, .28, .22, .22], bbox=[0, 0, 1, 1])
    tab.auto_set_font_size(False)
    tab.set_fontsize(16)
    for (row, _), cell in tab.get_celld().items():
        cell.set_facecolor("#18354a" if row else "#285168")
        cell.set_edgecolor(BACKGROUND)
        cell.get_text().set_color(WHITE)
    fig.text(.07, .22, "Шкала 1–5. Правильная разметка этого примера не задана.", color=WHITE, fontsize=18)
    fig.text(.07, .14, "Знакомые слова не означают уверенность в прогнозе.\nПримеры помогают увидеть ограничения и ошибки модели.", color=MUTED, fontsize=17)
    figures.append(fig)

    fig = base(5, "Проверим вашу ML-задачу")
    lines(fig, ["Одна задача. Один набор данных. Согласованная метрика.",
                "Проверка утечек, сравнение моделей, разбор ошибок.",
                "Результат: воспроизводимый ноутбук и короткий отчёт."])
    figures.append(fig)

    paths = []
    for index, figure in enumerate(figures, 1):
        path = frames / f"frame-{index}.png"
        figure.savefig(path, facecolor=BACKGROUND, dpi=100)
        plt.close(figure)
        paths.append(path)
    manifest = frames / "frames.txt"
    manifest.write_text("".join(f"file '{path}'\nduration 12\n" for path in paths) + f"file '{paths[-1]}'\n")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
                    "-i", str(manifest), "-t", "60", "-r", "12", "-c:v", "libx264", "-threads", "2",
                    "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    str(output)], check=True)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    build(args.bundle, args.output)
