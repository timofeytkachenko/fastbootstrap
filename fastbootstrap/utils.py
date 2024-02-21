from IPython.display import display, Markdown


def display_markdown_cell_by_significance(p_value):
    if p_value < 0.05:
        display(Markdown(
            '''<div class="alert alert-block alert-success">Difference are significant($p$_$value$ < 0.05)</div>''')
        )
    else:
        display(Markdown(
            '''<div class="alert alert-block alert-danger">Difference are non-significant($p$_$value$ > 0.05)</div>''')
        )
