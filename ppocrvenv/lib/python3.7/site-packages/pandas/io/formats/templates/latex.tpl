{% if parse_wrap(table_styles, caption) %}
\begin{table}
{%- set position = parse_table(table_styles, 'position') %}
{%- if position is not none %}
[{{position}}]
{%- endif %}

{% set position_float = parse_table(table_styles, 'position_float') %}
{% if position_float is not none%}
\{{position_float}}
{% endif %}
{% if caption and caption is string %}
\caption{% raw %}{{% endraw %}{{caption}}{% raw %}}{% endraw %}

{% elif caption and caption is sequence %}
\caption[{{caption[1]}}]{% raw %}{{% endraw %}{{caption[0]}}{% raw %}}{% endraw %}

{% endif %}
{% for style in table_styles %}
{% if style['selector'] not in ['position', 'position_float', 'caption', 'toprule', 'midrule', 'bottomrule', 'column_format'] %}
\{{style['selector']}}{{parse_table(table_styles, style['selector'])}}
{% endif %}
{% endfor %}
{% endif %}
\begin{tabular}
{%- set column_format = parse_table(table_styles, 'column_format') %}
{% raw %}{{% endraw %}{{column_format}}{% raw %}}{% endraw %}

{% set toprule = parse_table(table_styles, 'toprule') %}
{% if toprule is not none %}
\{{toprule}}
{% endif %}
{% for row in head %}
{% for c in row %}{%- if not loop.first %} & {% endif %}{{parse_header(c, multirow_align, multicol_align, True)}}{% endfor %} \\
{% endfor %}
{% set midrule = parse_table(table_styles, 'midrule') %}
{% if midrule is not none %}
\{{midrule}}
{% endif %}
{% for row in body %}
{% for c in row %}{% if not loop.first %} & {% endif %}
  {%- if c.type == 'th' %}{{parse_header(c, multirow_align, multicol_align)}}{% else %}{{parse_cell(c.cellstyle, c.display_value, convert_css)}}{% endif %}
{%- endfor %} \\
{% endfor %}
{% set bottomrule = parse_table(table_styles, 'bottomrule') %}
{% if bottomrule is not none %}
\{{bottomrule}}
{% endif %}
\end{tabular}
{% if parse_wrap(table_styles, caption) %}
\end{table}
{% endif %}
