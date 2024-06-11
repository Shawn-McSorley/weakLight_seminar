{% extends 'latex.tplx' %}

{% block input_group %}
{% if 'keep_input' in cell.metadata %}
{{ super() }}
{% else %}
\begin{flushleft}
{{ cell.source | escape_latex }}
\end{flushleft}
{% endif %}
{% endblock input_group %}
