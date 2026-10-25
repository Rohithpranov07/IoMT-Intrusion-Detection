"""Cohen's kappa for the free-text coding of 'which alerts looked less reliable'.

Each coder, independently and without seeing the other's codes, marks every answer:
yes = names the UNVERIFIED status line (or removal barely changing the decision) as the reason,
partly = mentions unreliability but not the status line, no = neither.
Usage: python3 reader_study/kappa_free_text.py free_text_coding_sheet.csv
"""
import sys
import pandas as pd
from sklearn.metrics import cohen_kappa_score

df = pd.read_csv(sys.argv[1], keep_default_na=False)
a = df.filter(like="coder_A").iloc[:, 0].str.strip().str.lower()
b = df.filter(like="coder_B").iloc[:, 0].str.strip().str.lower()
ok = (a != "") & (b != "")
print(f"answers coded by both: {ok.sum()} of {len(df)}")
print(f"Cohen's kappa: {cohen_kappa_score(a[ok], b[ok]):.2f}")
agree = a[ok] == b[ok]
print(f"raw agreement: {agree.mean():.0%}")
print("spotted the status line (both coders 'yes'):", int(((a == "yes") & (b == "yes")).sum()), "of", len(df))
