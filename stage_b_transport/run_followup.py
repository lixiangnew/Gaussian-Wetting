"""第一轮诊断后的补充：有效时间容差、干端正则化敏感性。

单独运行避免重复已通过的空间收敛实验。输出 followup_report.json。
"""
from pathlib import Path
import json
import numpy as np
from run_experiments import wetting_case


def main():
    out = Path(__file__).parent/'outputs'
    if not (out/'report.json').exists():
        raise RuntimeError('Run run_experiments.py first')
    original = json.loads((out/'report.json').read_text(encoding='utf-8'))
    with np.load(out/'top_state.npz',allow_pickle=False) as data:
        base_s = data['saturation'].copy()
    refined, refined_s = wetting_case(out,'tighter_tolerance',tolerance=2.5e-5)
    dry, _ = wetting_case(out,'dry_eps_002',initial_s=0,end_time=5,eps=0.02)
    dry_half, _ = wetting_case(out,'dry_eps_001',initial_s=0,end_time=5,eps=0.01)
    base = original['wetting']['top']['absorbed_kg']
    results = {
        'tighter_tolerance':refined,
        'time_accuracy':{'relative_absorption_difference':abs(refined['absorbed_kg']-base)/refined['absorbed_kg'],
                         's_l2_difference':float(np.sqrt(np.mean((base_s-refined_s)**2)))},
        'dry_eps_002':dry,'dry_eps_001':dry_half,
        'regularization_relative_absorption_difference':abs(dry['absorbed_kg']-dry_half['absorbed_kg'])/max(dry_half['absorbed_kg'],1e-15),
        'status':'completed',
    }
    (out/'followup_report.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(results,indent=2),flush=True)


if __name__ == '__main__':
    main()
