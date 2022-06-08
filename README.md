{% assign sorted = site.trials | sort: 'errorrate' %}
{% for trial in sorted %}
  <ul>
    <li>{{ trial.name }} ({{ trial.datetime }}) [{{ trial.errorrate }}]</li>
    <ul>
      <li>{{ trial.content | markdownify }}</li>
    </ul>
  </ul>
{% endfor %}
