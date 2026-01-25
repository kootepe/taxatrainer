# Taxa treenaus

Sovellus pyörii täällä:

[latu.ojakastikka.fi](https://latu.ojakastikka.fi)

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

docker-compose.prod.yml pyöräyttää sovelluksen docker konttiin gunicornin
taakse. Tämän avulla se pyörii tuolla latu.ojakastikka.fi.

Koska homma oli yksinkertainen, koodi on tehty 99% tekoälyllä eikä esim. nappien
sijoitteluun yms. ole käytetty juurikaan aikaa. Lajilistat olen koonnut
kurssimateriaalista käsin.
