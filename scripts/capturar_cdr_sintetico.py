"""Capture the unmodified CdR GUI with fictional records only.

Run from the thesis worktree. The CdR checkout is imported read-only.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cdr", type=Path, default=Path("D:/git/Comparador-de-Registros"))
    args = parser.parse_args()
    directory = Path(os.environ["TEMP"]) / "cdr_sint"
    directory.mkdir(parents=True, exist_ok=True)
    names = ["MARIA", "JOAO", "ANA", "PEDRO", "LUCIA", "PAULO", "ROSA", "LUIS", "CLARA", "BRUNO"]
    rows = [{"NOME,N": f"{name} EXEMPLO DA SILVA", "DTNASC,D": f"{i:02d}/01/1980"}
            for i, name in enumerate(names, 1)]
    comparison = [dict(row) for row in rows]
    comparison[1]["NOME,N"] = "JOAO EXEMPLO SILVA"
    comparison[2]["DTNASC,D"] = "04/01/1980"
    for filename, records in [("referencia_sintetica.csv", rows), ("comparacao_sintetica.csv", comparison)]:
        with (directory / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]), delimiter="|")
            writer.writeheader()
            writer.writerows(records)
    pairs = [{**{f"R_{key}": value for key, value in left.items()},
              **{f"C_{key}": value for key, value in right.items()}}
             for left, right in zip(rows, comparison)]
    input_path = directory / "pares_sinteticos.csv"
    with input_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]), delimiter="|")
        writer.writeheader()
        writer.writerows(pairs)

    sys.path.insert(0, str(args.cdr / "src"))
    # Suppress Python bytecode writes in the read-only CdR checkout.
    sys.dont_write_bytecode = True
    import gui
    import pyautogui

    app = gui.App()
    app.geometry("1150x620+80+80")
    app.filepath = str(input_path)
    app.e_in.insert(0, input_path.name)
    app._load_header()
    app._add_field()
    for widgets, field, kind in zip(app.boxes, ["NOME", "DTNASC"], ["N", "D"]):
        widgets["cb1"].set(app.left_labels[field])
        widgets["cb2"].set(app.right_labels[field])
        widgets["tipo_var"].set(gui.TIPO_LABELS[kind])
    app.e_out.delete(0, "end")
    app.e_out.insert(0, "resultado_sintetico")
    app.workers_var.set(1)
    app._update_sort_options()

    def capture() -> None:
        app.update_idletasks()
        region = (app.winfo_rootx(), app.winfo_rooty(), app.winfo_width(), app.winfo_height())
        image = pyautogui.screenshot(region=region)
        image.thumbnail((1600, 1600))
        destination = Path(__file__).resolve().parents[1] / "figures" / "fig_cdr_gui_mapeamento.png"
        image.save(destination)
        print(f"Screenshot: {destination}, size={image.size}", flush=True)
        app.e_out.delete(0, "end")
        app.e_out.insert(0, str(directory / "resultado_sintetico"))
        app._comparar()

    app.lift()
    app.attributes("-topmost", True)
    app.after(2500, capture)
    app.mainloop()


if __name__ == "__main__":
    main()
