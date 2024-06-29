import pandas as pd
import numpy as np

pd.set_option('precision', 4)

temp_df = pd.read_csv("raw_data.csv")
temp_df['temp_deg_C'] = temp_df['Temp']

temp_df['date'] = pd.to_datetime(temp_df['Date']) 
temp_df['year'] = [int(a.year) for a in temp_df['date']]
temp_df['years_since_19850101'] = (
	np.asarray([float((date - pd.to_datetime('1985-01-01')).days)
		for date in temp_df['date'].values])
	) / 365.0

train_years = [1981, 1982, 1983, 1984, 1985, 1986]
valid_years = [1987, 1988]
test_years = [1989, 1990]
df_by_name = dict()
for (name, years) in [
		('train_ByYear', train_years),
		('valid_ByYear', valid_years),
		('test_ByYear', test_years)]:
	bmask_N = temp_df['year'].isin(years)

	split_df = temp_df.loc[bmask_N][['date', 'years_since_19850101', 'temp_deg_C']].copy()
	split_df.to_csv("data_%s.csv" % name, index=False)
	df_by_name[name] = split_df

test_bmask_N = temp_df['year'].isin(test_years)
keep_bmask_N = np.logical_not(test_bmask_N)

# Select random 
n_valid = df_by_name['valid_ByYear'].shape[0]
valid_rows = np.random.RandomState(0).choice(np.flatnonzero(keep_bmask_N), n_valid, replace=False)
valid_bmask_N = np.zeros_like(keep_bmask_N)
valid_bmask_N[valid_rows] = 1

split_df = temp_df.loc[valid_bmask_N][['date', 'years_since_19850101', 'temp_deg_C']].copy()
split_df.to_csv("data_%s.csv" % ('valid_ByRandom'), index=False)

train_bmask_N = np.logical_and(keep_bmask_N, 1 - valid_bmask_N)
split_df = temp_df.loc[train_bmask_N][['date', 'years_since_19850101', 'temp_deg_C']].copy()
split_df.to_csv("data_%s.csv" % ('train_ByRandom'), index=False)


