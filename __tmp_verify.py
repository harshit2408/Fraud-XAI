import numpy as np, pandas as pd

# Verify card-aggregate variance formula
amounts = pd.Series([10.0, 20.0, 30.0])
card = pd.Series(['A','A','A'])
df = pd.DataFrame({'card1': card, 'TransactionAmt': amounts})
df['tc'] = df.groupby('card1').cumcount()
df['ts'] = df.groupby('card1')['TransactionAmt'].cumsum() - df['TransactionAmt']
ssq_shifted = (df['TransactionAmt']**2).groupby(df['card1']).shift(1)
ssq = df.assign(_sq=ssq_shifted).groupby('card1')['_sq'].cumsum().fillna(0.0)
var = np.where(df['tc'] > 1, (ssq - (df['ts']**2)/df['tc'])/(df['tc']-1), 0.0)

print('tc:', df['tc'].tolist())
print('ts:', df['ts'].tolist())
print('ssq:', ssq.tolist())
print('var:', var.tolist())
print('expected var for [10,20] (ddof=1):', np.var([10.0,20.0], ddof=1))
print('expected var for [10,20,30] (ddof=1):', np.var([10.0,20.0,30.0], ddof=1))

# Verify target encoding: is current row excluded in seeded path?
# carried_sum is a per-row array seeded from previous splits
# cum_sum = groupby(col)[target].cumsum() - target + carried_sum
# cumsum() includes current row, so subtracting target correctly excludes it
# carried_sum is looked up from state BEFORE this frame, so no future contamination
print()
print("Target encoding row-exclusion check:")
df2 = pd.DataFrame({'card1': [1,1,1], 'isFraud': [0,1,0]})
grouped = df2.groupby('card1')['isFraud']
# carried_sum=0 for all rows (no prior history)
carried_sum = np.zeros(3)
cum_sum = grouped.cumsum() - df2['isFraud'] + carried_sum
cum_count = grouped.cumcount() + np.zeros(3)
print('cum_sum:', cum_sum.tolist(), '(should be [0,0,1])')
print('cum_count:', cum_count.tolist(), '(should be [0,1,2])')

# Check what happens with non-zero carried_sum — is it a scalar or array?
# _carried_totals returns arrays of shape (len(keys),), one value per row
# (each row gets its entity's historical total)
# So carried_sum is shape (n,) — correct usage in cum_sum formula
print()
print("carried_sum shape test:")
keys = pd.Series([1, 2, 1])
carried = {1: (3.0, 2.0), 2: (1.0, 1.0)}
sum_map = {k: v[0] for k, v in carried.items()}
count_map = {k: v[1] for k, v in carried.items()}
cs = keys.map(sum_map).fillna(0.0).to_numpy(dtype=float)
cc = keys.map(count_map).fillna(0.0).to_numpy(dtype=float)
print('carried_sum:', cs, '(should be [3.0, 1.0, 3.0])')
print('carried_count:', cc, '(should be [2.0, 1.0, 2.0])')

# Verify the IPCA _partial_fit_batches edge case
from src.data.feature_engineering import _partial_fit_batches
import numpy as np
# Case: n=12, batch_size=10, min_batch=5
positions = np.arange(12)
batches = list(_partial_fit_batches(positions, 10, 5))
print()
print('_partial_fit_batches(12, 10, min=5):', [len(b) for b in batches], '(should merge: [12])')
# Case: n=22, batch_size=10, min_batch=5 => starts=[0,10], trailing=2 < 5 => fold into [0..22]
positions2 = np.arange(22)
batches2 = list(_partial_fit_batches(positions2, 10, 5))
print('_partial_fit_batches(22, 10, min=5):', [len(b) for b in batches2], '(should be [20,2]? no folding since 2<5)')
# starts=[0,10,20], n-20=2 < min_batch=5, so pop last => starts=[0,10], end=22 => [10,12]
print('--- expected [10, 12] (fold trailing 2 into second batch)')
