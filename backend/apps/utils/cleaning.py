import re
from typing import List


def normalize_cell(x):
    if isinstance(x, (bytes, bytearray)):
        try:
            x = x.decode("utf-8", errors="ignore")
        except:
            x = str(x)
    if isinstance(x, str):
        x = x.strip()
        m = re.match(r"^b[\"'](.*)[\"']$", x)
        if m:
            x = m.group(1)
        if (len(x) >= 2) and ((x[0] == x[-1]) and x[0] in ("'", '"')):
            x = x[1:-1]
    return x


def clean_dataframe_bytes(df, cols=None):
    if cols is None:
        cols = [c for c in df.columns if df[c].dtype == object]
    for c in cols:
        df[c] = df[c].apply(normalize_cell)
    return df


def parse_genres(g) -> List[str]:
    s = str(g).strip().strip("[]")
    if s == "":
        return ['0']
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
    else:
        parts = [p.strip() for p in s.split() if p.strip()]
    return parts if parts else ['0']
