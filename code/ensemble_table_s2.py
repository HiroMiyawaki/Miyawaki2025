# %%
from pathlib import Path
import re
from datetime import datetime
from my_tools.color_preset import merge_xmlx_sheets


def get_latest_sorted_xlsx_files(stats_dir):
    """
    指定ディレクトリ内の .xlsx ファイルのうち、最新の fig_n および fig_sn のファイルを取得し、
    fig_n → fig_sn の順にソートしてリスト化する。

    :param stats_dir: (Path or str) .xlsx ファイルが含まれるディレクトリ
    :return: (list of Path) ソートされた最新の .xlsx ファイルのリスト
    """
    stats_dir = Path(stats_dir)  # Path オブジェクト化
    xlsx_files = list(stats_dir.rglob("*.xlsx"))  # .xlsx ファイルを再帰的に取得

    # 最新ファイルを格納する辞書
    latest_files = {}
    newest_date = datetime(1900, 1, 1)

    # 正規表現パターン
    pattern = re.compile(r"(fig_s?\d+)_(\d{4}_\d{4})\.xlsx")

    for file in xlsx_files:
        match = pattern.match(file.name)
        if match:
            key = match.group(1)  # fig_n または fig_sn
            date_str = match.group(2)  # yyyy_mmdd
            file_date = datetime.strptime(date_str, "%Y_%m%d")  # 日付に変換

            # 最新ファイルの更新
            if key not in latest_files or file_date > latest_files[key][1]:
                latest_files[key] = (file, file_date)

                if file_date > newest_date:
                    newest_date = file_date

    # カスタムソート関数
    def custom_sort_key(key):
        match = re.match(r"fig_?(s?)(\d+)", key)
        if match:
            is_s = match.group(1) == "s"  # `s` の有無
            number = int(match.group(2))  # 数値部分を取得
            return (is_s, number)  # `s` の有無でソート、次に数字でソート
        return (True, float('inf'))

    # 最新ファイルのキーをソート
    sorted_keys = sorted(latest_files.keys(), key=custom_sort_key)

    # ソートされた最新ファイルリストを作成
    sorted_files = [latest_files[key][0] for key in sorted_keys]

    return sorted_files, newest_date


# 使用例
stats_dir = Path("~/Dropbox/coact_stability_paper/stats/").expanduser()
latest_sorted_files, newest_date = get_latest_sorted_xlsx_files(stats_dir)

output_file = Path(
    f"~/Dropbox/coact_stability_paper/table_s2_{newest_date.strftime('%Y_%m%d')}.xlsx").expanduser()
merge_xmlx_sheets(latest_sorted_files, output_file)

# %%
