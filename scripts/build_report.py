"""Build the technical report PDF from Report/report_source.html using headless Edge or Chrome.

Test sentences come from sentences.json, and plots and training records come from real files in
Results/. While any section is still marked pending, the output is a draft under output/, so an
unfinished report cannot land in Report/Technical_Report.pdf.
"""
import csv
import html
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Report" / "report_source.html"
BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "msedge", "google-chrome", "chromium",
]
PENDING = '<div class="pending">{}</div>'


def image(path, caption):
    return '<figure><img src="{}"><figcaption>{}</figcaption></figure>'.format(path.as_uri(), html.escape(caption))


def loss_plots():
    parts = []
    for language in ("English", "Telugu"):
        for kind in ("training", "validation"):
            path = ROOT / "Results" / language / "{}_loss.png".format(kind)
            parts.append(image(path, "{} {} loss".format(language, kind)) if path.exists()
                         else PENDING.format("{} {} loss plot".format(language, kind)))
    return "\n".join(parts)


def variance_plots():
    sentences = json.loads((ROOT / "data" / "test_utterances" / "sentences.json").read_text(encoding="utf-8"))
    parts = []
    for language in ("english", "telugu"):
        name = language.capitalize()
        for sentence in sentences[language]:
            for kind in ("duration", "pitch", "energy"):
                path = ROOT / "Results" / name / "variance_adaptor_plots" / kind / (sentence["id"] + ".png")
                parts.append(image(path, "{} {}".format(sentence["id"], kind)) if path.exists()
                             else PENDING.format("{} {} plot".format(sentence["id"], kind)))
    return "\n".join(parts)


def training_record():
    rows = []
    for language in ("English", "Telugu"):
        path = ROOT / "Results" / language / "train_loss.csv"
        if not path.exists():
            rows.append(PENDING.format("{} steps reached and training time".format(language)))
            continue
        with open(path, encoding="utf-8", newline="") as f:
            records = list(csv.DictReader(f))
        seconds = sum(float(r["seconds_since_log"] or 0) for r in records)
        rows.append("<p>{}: {} steps logged, about {:.1f} h of recorded training time.</p>".format(
            language, records[-1]["step"], seconds / 3600))
    return "\n".join(rows)


def main():
    text = SOURCE.read_text(encoding="utf-8")
    sentences = json.loads((ROOT / "data" / "test_utterances" / "sentences.json").read_text(encoding="utf-8"))
    for language in ("english", "telugu"):
        for sentence in sentences[language]:
            text = text.replace("{{" + sentence["id"] + "}}", html.escape(sentence["text"]).replace("  ", "&nbsp; "))
    text = text.replace("{{LOSS_PLOTS}}", loss_plots()).replace("{{VARIANCE_PLOTS}}", variance_plots())
    text = text.replace("{{TRAINING_RECORD}}", training_record())
    pending = text.count('class="pending"')
    if pending:
        status = "<b>DRAFT, not for submission: {} items await trained models and listening.</b>".format(pending)
        out = ROOT / "output" / "Technical_Report_DRAFT.pdf"
    else:
        status = "Assignment for Intern Position at LTRC, IIIT Hyderabad"
        out = ROOT / "Report" / "Technical_Report.pdf"
    text = text.replace("{{STATUS_LINE}}", status)
    built = ROOT / "output" / "report_built.html"
    built.write_text(text, encoding="utf-8")
    browser = next((b for b in BROWSERS if os.path.exists(b) or shutil.which(b)), None)
    if browser is None:
        raise RuntimeError("Edge or Chrome is needed to print the report")
    subprocess.run(
        [browser, "--headless", "--disable-gpu", "--no-pdf-header-footer", "--print-to-pdf=" + str(out), built.as_uri()],
        check=True, capture_output=True, timeout=180,
    )
    print("Wrote", out, "with", pending, "pending items")


if __name__ == "__main__":
    main()
