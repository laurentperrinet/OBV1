name = 'thesis'

import nbconvert
import nbformat

nb_list = []
import glob
for fname in glob.glob('*.ipynb'):
    if fname[0] in ['1', '2',  '3', '4']:
        print ("'{}', ".format(fname) )
        nb_list.append(fname)
        
def strip(nb):
    """
    Keeps only the cells :
    - starting with the first to begin with a section (that is with a ``#``)
    - stoping with the next cell to begin with a section (that is with a ``#``)
    
    """
    start, stop = -1, len(nb.cells)
    nb_out = nb.copy()
    for i_cell, cell in enumerate(nb_out.cells):
        if len(cell['source'])>0:
            if cell['source'][0] == '#':
                if start == -1: start = i_cell
                else:
                    if stop == len(nb.cells): stop = i_cell
    if start == -1: start = 0
    nb_out.cells = nb.cells[start:stop]
    return nb_out
    
def merge_notebooks(outfile, filenames):
    merged = None
    for fname in filenames:
        with open(fname, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, nbformat.NO_CONVERT)
        
        nb = strip(nb)
        if merged is None:
            merged = nb
        else:
            merged.cells.extend(nb.cells)
    with open(outfile, 'w', encoding='utf-8') as f:
        f.write(nbformat.writes(merged, nbformat.NO_CONVERT))
merge_notebooks(name + '.ipynb', nb_list)  

with open(name + '.ipynb', 'r') as f:
    nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)

latex_exporter = nbconvert.PDFExporter()
latex_exporter.template_file = name # assumes it has the same name as the output
latex_exporter.verbose = True
(body, resources) = latex_exporter.from_notebook_node(nb)
with open(name + '.pdf', 'w', encoding="iso-8859-1") as f:
    f.write(body.decode(encoding="iso-8859-1"))