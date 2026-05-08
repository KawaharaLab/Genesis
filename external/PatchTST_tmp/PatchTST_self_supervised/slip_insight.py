import pandas as pd
from pathlib import Path

INPUT_CSV = Path('/home/user/PatchTST/data/gpt/predictions_wo_0 copy.csv')
# INPUT_CSV = Path('/home/user/PatchTST/data/gpt/predictions_wo_0.csv')
OUTPUT_CSV = INPUT_CSV.parent / 'slip_insight_summary.csv'

SLIP_LABEL = 'slip'  # ground-truth / prediction value indicating slip

def analyze(df: pd.DataFrame) -> pd.DataFrame:
	# Ensure expected columns
	required = {'csv_path', 'start', 'true', 'pred'}
	missing = required - set(df.columns)
	if missing:
		raise ValueError(f'Missing columns in input CSV: {missing}')

	# Convert start to numeric just in case
	df['start'] = pd.to_numeric(df['start'], errors='coerce')

	rows = []
	for csv_path, g in df.groupby('csv_path'):
		g_sorted = g.sort_values('start')

		# First true slip (must exist; otherwise we skip this csv_path)
		true_slip_rows = g_sorted[g_sorted['true'] == SLIP_LABEL]
		if true_slip_rows.empty:
			continue  # 真の slip が一度も無いデータは除外
		first_true = true_slip_rows['start'].iloc[0]
		pred_slip_rows = g_sorted[g_sorted['pred'] == SLIP_LABEL]
		first_pred = pred_slip_rows['start'].iloc[0] if not pred_slip_rows.empty else pd.NA
		# First time both true and pred are slip
		joint_rows = g_sorted[(g_sorted['true'] == SLIP_LABEL) & (g_sorted['pred'] == SLIP_LABEL)]
		first_joint = joint_rows['start'].iloc[0] if not joint_rows.empty else pd.NA

		rows.append({
			'csv_path': csv_path,
			'first_true_slip_start': first_true,
			'first_correct_pred_slip_start': first_joint,
			'first_pred_slip_start': first_pred
		})

	return pd.DataFrame(rows)
def remove_zero(df: pd.DataFrame) -> pd.DataFrame:
    return df[df['start'] != 0]

def compute_simple_accuracy(df: pd.DataFrame) -> dict:
	"""全体の単純精度を算出して返す。"""

	# 欠損を除外（true/pred どちらか欠けた行は精度計算から外す）
	df = df[df['start'] != 0]
	valid = df.dropna(subset=['true', 'pred'])
	total = int(len(valid))
	total_slip = int((valid['true'] == SLIP_LABEL).sum())
	total_stable = total - total_slip
	correct_slip = int(((valid['true'] == SLIP_LABEL) & (valid['pred'] == SLIP_LABEL)).sum())
	correct_stable = int(((valid['true'] != SLIP_LABEL) & (valid['pred'] != SLIP_LABEL)).sum())
	acc = (correct_slip + correct_stable) / total if total > 0 else float('nan')
	print(f"'overall_accuracy': {acc}, 'total': {total}, 'correct_slip': {correct_slip}/{total_slip}, 'correct_stable': {correct_stable}/{total_stable}")

def main():
	df = pd.read_csv(INPUT_CSV)
	metrics = compute_simple_accuracy(df)


# def main():
# 	df = pd.read_csv(INPUT_CSV)
# 	summary = analyze(df)
# 	summary.to_csv(OUTPUT_CSV, index=False)
# 	print(f'Wrote summary to {OUTPUT_CSV}')
# 	print(summary.head())


if __name__ == '__main__':
	# remove_zero(pd.read_csv(INPUT_CSV)).to_csv(INPUT_CSV, index=False)
	main()