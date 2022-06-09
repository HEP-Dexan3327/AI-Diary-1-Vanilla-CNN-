{% assign sorted = site.trials | sort: 'errorrate' %}
{% for trial in sorted reversed %}
  <ul>
    <li>{{ trial.name }} <span style="color:orange;">({{ trial.datetime }})</span> [ Accuracy: {{ trial.errorrate }}]</li>
    <ul>
      <li><img src="models/{{ trial.img }}" alt="Model Structure: ">{{ trial.content | markdownify }}</li>
    </ul>
  </ul>
{% endfor %}
