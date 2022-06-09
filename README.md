{% assign sorted = site.trials | sort: 'errorrate' %}
{% for trial in sorted %}
  <ul>
    <li>{{ trial.name }} <span style="color:orange;">({{ trial.datetime }})</span> [ Accuracy: {{ trial.errorrate }}]</li>
    <ul>
      <li><img src="{{ trial.img }}"></li>
      <li>{{ trial.content | markdownify }}</li>
    </ul>
  </ul>
{% endfor %}
