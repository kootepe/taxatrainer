# Taxa treenaus

Sovellus UEF:in selkärangattomien ja selkärankaisten lajituntemuskurssin tieteellisten nimien harjoitteluun.

Lajilistat on kopioitu materiaaleista joulukuussa 2025.

Sovelluksen voi ajaa omalla koneellakin kopioimalla repon ja ajamalla nämä
komennot.

```
# tehdään python virtualenv
python3 -m venv .env
# aktivoidaan env
source .env/bin/activate
# asennetaan flask
pip install flask
# ajetaan varsinainen ohjelma
python3 app.py
```

docker-compose.prod.yml pyöräyttää sovelluksen docker konttiin gunicorning
taakse.
