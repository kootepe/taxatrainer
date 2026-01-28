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
# jos olet windowsilla, googlaa miten aktivoida python env
# aktivoidaan env
source .env/bin/activate
# asennetaan flask
pip install flask
# aja yksi seuraavista riippuen käyttöjärjestelmästä
# jos COOKIE_SECURE on TRUE/1 (oletus) niin et voi tallentaa asetuksia
# jos olet linuxilla
COOKIE_SECURE=0 python3 app.py
# jos olet windowsilla powershellissä
$env:COOKIE_SECURE=0; python app.py
# jos olet windowsilla cmdssä
set COOKIE_SECURE=0 && python app.py
```

docker-compose.prod.yml pyöräyttää sovelluksen docker konttiin gunicornin
taakse. Tämän avulla se pyörii tuolla latu.ojakastikka.fi.

Koska homma oli yksinkertainen, koodi on tehty 99% tekoälyllä eikä esim. nappien
sijoitteluun yms. ole käytetty juurikaan aikaa. Lajilistat olen koonnut
kurssimateriaalista käsin.
