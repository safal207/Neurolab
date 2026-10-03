"""Notebook demo using the saved text pipeline and all trained comparisons."""

from __future__ import annotations

import html
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .experiment import AXES, load_predictor, neural_predict


EXAMPLES = {
    "Хорошая новость": "The team solved the problem and I am happy with the result.",
    "Срыв срока": "The payment failed again and I am worried about the deadline.",
    "Спокойный день": "I feel calm and relaxed after a quiet day at home.",
}


class DemoPredictor:
    """Load locally trusted lab artifacts once; do not train or select on demo text."""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.report = json.loads((self.output_dir / "report.json").read_text())
        self.features = joblib.load(self.output_dir / "features.joblib")
        self.ridge = joblib.load(self.output_dir / "ridge.joblib")
        self.models = {k: load_predictor(self.output_dir, k=k)[0]
                       for k in self.report["config"]["iterations"]}

    def inspect_text(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Введите непустой английский текст.")
        if len(text) > 2000:
            raise ValueError("Для демо используйте текст до 2 000 символов.")
        vectorizer = self.features[0]
        tokens = vectorizer.build_analyzer()(text)
        matched = sum(token in vectorizer.vocabulary_ for token in tokens)
        if matched == 0:
            raise ValueError("В тексте нет признаков из обученного словаря. Попробуйте английское предложение из примеров.")
        return {"features": len(tokens), "matched": matched, "coverage": matched / len(tokens)}

    def predict(self, texts):
        texts = list(texts)
        if not texts:
            raise ValueError("Добавьте хотя бы один текст.")
        coverage = [self.inspect_text(text) for text in texts]
        inputs = self.features.transform(texts).astype(np.float32)
        predictions = {"Ridge": np.clip(self.ridge.predict(inputs), 1, 5)}
        predictions.update({f"Neurolab K={k}": np.clip(neural_predict(model, inputs, k), 1, 5)
                            for k, model in self.models.items()})
        rows = [{"text": text, "method": method, **dict(zip(AXES, values[i]))}
                for i, text in enumerate(texts) for method, values in predictions.items()]
        table = pd.DataFrame(rows)
        table.attrs["coverage"] = coverage
        return table

    def benchmark(self):
        return pd.DataFrame(self.report["test_metrics"])[["model", "MAE_mean"]]


def result_html(table, predictor):
    """Escape all user text before rendering a compact, portable HTML result."""
    text = html.escape(str(table.iloc[0]["text"]))
    coverage = table.attrs["coverage"][0]
    rows = "".join(
        "<tr><td>" + html.escape(row["method"]) + "</td>"
        + "".join(f"<td>{float(row[axis]):.3f}</td>" for axis in AXES) + "</tr>"
        for _, row in table.iterrows())
    benchmark_rows = "".join(
        f"<tr><td>{html.escape(row['model'])}</td><td>{float(row['MAE_mean']):.4f}</td></tr>"
        for _, row in predictor.benchmark().iterrows())
    return f"""<div style="background:#f5f9fc;color:#183249;padding:18px;border-radius:12px;max-width:760px;line-height:1.55">
    <strong style="font-size:18px">Сравнение сохранённых моделей</strong><p>{text}</p>
    <table style="width:100%;text-align:left;color:#183249">
    <thead><tr><th>Метод</th><th>V · положительность</th><th>A · активность</th><th>D · контроль</th></tr></thead>
    <tbody>{rows}</tbody></table>
    <p>Шкала 1–5. Оценки относятся к тексту. Правильный ответ для этого примера не задан.</p>
    <p>Совпало признаков со словарём: {coverage['matched']} из {coverage['features']} ({coverage['coverage']:.0%}).
    Это покрытие словаря, а не уверенность в прогнозе. Язык автоматически не определяется.</p>
    <details><summary>Ошибка на отложенных текстах EmoBank · меньше лучше</summary>
    <table style="color:#183249;text-align:left"><thead><tr><th>Метод</th><th>Средняя MAE</th></tr></thead>
    <tbody>{benchmark_rows}</tbody></table>
    <p>Это результат сохранённого эксперимента, а не ошибка введённого текста.</p></details>
    </div>"""


def create_demo(output_dir):
    """Return a widget panel; works offline after the bundle is trained."""
    import ipywidgets as widgets
    from IPython.display import HTML, clear_output, display

    predictor = DemoPredictor(output_dir)
    text = widgets.Textarea(value=EXAMPLES["Хорошая новость"],
                            placeholder="Введите английский текст…",
                            layout=widgets.Layout(width="100%", height="100px"))
    run = widgets.Button(description="Сравнить модели", button_style="primary", icon="play")
    output = widgets.Output(layout=widgets.Layout(width="100%"))

    def render(_=None):
        with output:
            clear_output(wait=True)
            try:
                table = predictor.predict([text.value])
            except ValueError as error:
                display(HTML(f"<p role='alert'>{html.escape(str(error))}</p>"))
            else:
                display(HTML(result_html(table, predictor)))

    buttons = []
    for title, sentence in EXAMPLES.items():
        button = widgets.Button(description=title)
        def choose(_, value=sentence):
            text.value = value
            render()
        button.on_click(choose)
        buttons.append(button)
    run.on_click(render)
    panel = widgets.VBox([
        widgets.HTML("<h3>Neurolab · один текст, три метода</h3>"
                     "<p>Выберите пример или введите свой английский текст. Модели уже обучены.</p>"),
        widgets.HBox(buttons, layout=widgets.Layout(flex_flow="row wrap")), text, run, output,
    ], layout=widgets.Layout(width="100%", max_width="800px"))
    render()
    return panel
