{% assign trials = site.trials | group_by: 'dataset' %}
{% for group in trials %}
  <h1>HI</h1>
  {% assign sorted = item.trials | sort: 'errorrate' | reverse %}
  {% for trial in sorted  %}
   <ul>
    <li>{{ trial.name }} <span style="color:orange;">({{ trial.datetime }})</span> [ Accuracy: {{ trial.errorrate }}]</li>
    <ul>
      <li><img src="models/{{ trial.img }}" alt="Model Structure: ">{{ trial.content | markdownify }}</li>
    </ul>
  </ul>
  {% endfor %}
{% endfor %}
