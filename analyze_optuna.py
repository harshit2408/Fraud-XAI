import optuna
import pandas as pd

study = optuna.load_study(study_name='tft_extensive_tuning', storage='sqlite:///tft_tuning.db')

trials = study.trials
completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
pruned = [t for t in trials if t.state == optuna.trial.TrialState.PRUNED]

with open('optuna_analysis.txt', 'w') as f:
    f.write(f'Total Trials: {len(trials)}\n')
    f.write(f'Completed: {len(completed)}\n')
    f.write(f'Pruned: {len(pruned)}\n')
    f.write('\n--- Best 5 Trials ---\n')
    completed.sort(key=lambda t: t.value, reverse=True)
    for i, t in enumerate(completed[:5]):
        f.write(f'Rank {i+1}: Trial {t.number}, PR-AUC: {t.value:.4f}\n')
        f.write(f'  Params: {t.params}\n')

    f.write('\n--- Top 10 Parameter Frequency ---\n')
    top_10 = completed[:10]
    if len(top_10) > 0:
        params = [t.params for t in top_10]
        df = pd.DataFrame(params)
        for col in df.columns:
            if col != 'learning_rate':
                f.write(f'\n{col}:\n')
                f.write(df[col].value_counts().to_string() + '\n')
        f.write('\nlearning_rate summary:\n')
        f.write(df['learning_rate'].describe().to_string() + '\n')
