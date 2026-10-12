"""检查已生成输出并制作终态剖面对照，不重复运行物理实验。"""
from pathlib import Path
import ast
import csv
import json
import platform
import re
import xml.etree.ElementTree as ET
import numpy as np
from run_experiments import line_svg


def main():
    root = Path(__file__).resolve().parent
    out = root/'outputs'
    for filename in ('report.json','followup_report.json'):
        assert json.loads((out/filename).read_text())['status'] == 'completed'
    for path in root.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'))
    checked_links = 0
    for path in root.glob('*.md'):
        for link in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            link = link.strip('<>').split('#',1)[0]
            if not link or '://' in link:
                continue
            assert (path.parent/link).exists(),link
            checked_links += 1
    state_ranges = {}
    for path in out.glob('*_state.npz'):
        with np.load(path,allow_pickle=False) as data:
            for name in data.files:
                assert np.isfinite(data[name]).all(),(path,name)
            s = data['saturation']
            assert s.min() >= -1e-10 and s.max() <= 1+1e-10
            state_ranges[path.stem] = [float(s.min()),float(s.max())]
    max_residual = 0.0
    for path in out.glob('*_history.csv'):
        with path.open(newline='',encoding='utf-8') as f:
            previous = -1.0
            for row in csv.DictReader(f):
                assert float(row['time_s']) > previous
                previous = float(row['time_s'])
                max_residual = max(max_residual,abs(float(row['mass_residual_kg'])))
                assert float(row['min_s']) >= -1e-10 and float(row['max_s']) <= 1+1e-10
    for target,names in [('top_profiles',('top','surface_only','impermeable')),
                         ('bottom_profiles',('bottom_gravity','bottom_no_gravity'))]:
        series = []
        for name in names:
            with np.load(out/f'{name}_state.npz',allow_pickle=False) as data:
                z = np.unique(data['centers'][:,2])
                profile = [data['saturation'][np.isclose(data['centers'][:,2],v)].mean() for v in z]
            series.append((name,z,np.asarray(profile)))
        line_svg(out/f'{target}.svg',target,series,'height z (m)','S')
    svgs = list(out.glob('*.svg'))
    for path in svgs:
        ET.parse(path)
    result = {'python':platform.python_version(),'numpy':np.__version__,
              'checked_local_links':checked_links,'svg_files_parsed':len(svgs),
              'max_history_mass_residual_kg':max_residual,'state_ranges':state_ranges,
              'status':'passed','note':'SVG XML and data checked; not a screenshot-based layout review.'}
    (out/'output_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
