"""Summarize error sources in the cached 15 grouped validation runs."""
import json
from pathlib import Path
import numpy as np

z = np.load('artifacts/probability_oof.npz')
y = np.concatenate([z[f'{i}_y'] for i in range(15)])
p = np.concatenate([z[f'{i}_p'] for i in range(15)])
q = np.concatenate([z[f'{i}_q'] for i in range(15)])
pred = 10000*p + (1-p)*q
error = (y-pred)**2
report = {'model': 'uncalibrated text/syntax mixture',
          'observation_note': 'Rows can recur across grouped validation runs; counts are held-out observations, not unique molecules.',
          'pooled_mse': float(error.mean()),
          'mean_fold_mse': float(np.mean([np.mean((z[f'{i}_y']-(10000*z[f'{i}_p']+(1-z[f'{i}_p'])*z[f'{i}_q']))**2) for i in range(15)])),
          'groups': {}}
for name, mask in [('SI=10000', y == 10000), ('SI>10000', y > 10000), ('SI<10000', y < 10000)]:
    report['groups'][name] = {'heldout_observations': int(mask.sum()),
                             'squared_error_share': float(error[mask].sum()/error.sum()),
                             'mse': float(error[mask].mean())}
report['diagnostic_oracle_probability_mse'] = float(np.mean((y-((y == 10000)*10000+(y != 10000)*q))**2))
report['diagnostic_oracle_conditional_mse'] = float(np.mean((y-(p*10000+(1-p)*y))**2))
report['oracle_note'] = 'Diagnostics only; true validation targets are never available at prediction time.'
Path('validation_error_audit.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
