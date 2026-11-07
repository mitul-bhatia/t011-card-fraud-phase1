"""Run this phase's notebooks in fresh processes and export linked reading pages."""
from pathlib import Path
import os
import re
import subprocess
import sys
import time
import traceback
from urllib.parse import urlsplit, urlunsplit

PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE.parent

def reading_copy(book, path):
    from nbconvert import HTMLExporter
    output = PHASE / ('output/drafts/gmm_k_sweep' if path.parent.name == 'drafts'
                      else 'output/notebooks')
    output.mkdir(parents=True, exist_ok=True)
    target = output / (path.stem + '.html')
    pages = sorted(path.parent.glob('*.ipynb'))
    position = pages.index(path)
    home = os.path.relpath(ROOT/'index.html', output)
    links = [f'<a href="{home}#{PHASE.name}">Project home</a>']
    if position:
        links.append(f'<a href="{pages[position-1].stem}.html">Previous notebook</a>')
    if position + 1 < len(pages):
        links.append(f'<a href="{pages[position+1].stem}.html">Next notebook</a>')
    source = os.path.relpath(path, output)
    links.append(f'<a href="{source}">Download notebook</a>')
    navigation = ('<nav aria-label="Notebook navigation" style="display:flex;flex-wrap:wrap;'
                  'gap:12px 24px;max-width:1120px;margin:22px auto;padding:16px 24px;'
                  'border-bottom:1px solid #d5dfe4;font:16px/1.5 Arial,sans-serif">'
                  + ' '.join(links) + '</nav>')
    page, _ = HTMLExporter(embed_images=True).from_notebook_node(book)
    # Markdown paths are relative to the .ipynb file, not its HTML export.
    def rebase(match):
        attribute, value = match.group(1), match.group(2)
        url = urlsplit(value)
        if url.scheme or url.netloc or not url.path:
            return match.group(0)
        source = (path.parent / url.path).resolve()
        if not source.exists():
            return match.group(0)
        relative = os.path.relpath(source, target.parent)
        return f'{attribute}="{urlunsplit(("", "", relative, url.query, url.fragment))}"'
    page = re.sub(r'(href|src)="([^"]+)"', rebase, page)
    page = re.sub(r'(<body\b[^>]*>)', lambda m: m.group(1)+navigation, page, count=1)
    page = page.replace('</body>', navigation+'</body>')
    target.write_text(page)

def run_one(path):
    os.chdir(PHASE)
    cache = ROOT / '.cache' / PHASE.name
    os.environ['MPLCONFIGDIR'] = str(cache / 'matplotlib')
    os.environ['IPYTHONDIR'] = str(cache / 'ipython')
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['LOKY_MAX_CPU_COUNT'] = '2'
    import nbformat
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    shell = InteractiveShell.instance()
    book = nbformat.read(path, as_version=4)
    for cell in book.cells:
        if cell.cell_type == 'code':
            cell.execution_count, cell.outputs = None, []
    book.metadata['execution'] = {'all_code_cells_succeeded': False, 'fresh_process': True}
    nbformat.write(book, path)
    count, start = 0, time.perf_counter()
    for cell in book.cells:
        if cell.cell_type != 'code':
            continue
        count += 1
        print(f'{PHASE.name}/{path.name}: running code cell {count}', flush=True)
        with capture_output() as captured:
            result = shell.run_cell(cell.source, store_history=True)
        outputs = []
        for stream in ['stdout', 'stderr']:
            value = getattr(captured, stream)
            if value:
                outputs.append(nbformat.v4.new_output('stream', name=stream, text=value))
        for item in captured.outputs:
            outputs.append(nbformat.v4.new_output('display_data', data=item.data, metadata=item.metadata))
        error = result.error_before_exec or result.error_in_exec
        if error:
            outputs.append(nbformat.v4.new_output('error', ename=type(error).__name__,
                evalue=str(error), traceback=traceback.format_exception(error)))
        cell.execution_count, cell.outputs = count, outputs
        nbformat.write(book, path)
        if error:
            print(captured.stdout, captured.stderr)
            raise RuntimeError(f'{path.name}: code cell {count} failed') from error
    book.metadata['execution'] = {'all_code_cells_succeeded': True, 'fresh_process': True,
                                   'seconds': time.perf_counter()-start}
    nbformat.validate(book)
    nbformat.write(book, path)
    reading_copy(book, path)
    print(f'Completed {path.name} in {time.perf_counter()-start:.1f}s', flush=True)

if __name__ == '__main__':
    if len(sys.argv) == 2:
        run_one(Path(sys.argv[1]).resolve())
    else:
        for path in sorted((PHASE / 'notebooks').glob('*.ipynb')):
            subprocess.run([sys.executable, str(Path(__file__).resolve()), str(path)], check=True)
        print(f'All {PHASE.name} notebooks passed. Reading copies: {PHASE.name}/output/notebooks/.')
