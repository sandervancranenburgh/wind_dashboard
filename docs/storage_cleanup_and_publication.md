# Dagarchieven, opschoning en publicatie zonder Git-groei

## Status en uitgangspunten

Dit is een gefaseerde migratie. De wijzigingen zijn voorbereid op
`codex/storage-and-publication-20261016` in een geïsoleerde werkmap. Er zijn
geen productiearchieven verwijderd, geen Git-geschiedenis herschreven en geen
Pages-instellingen, crontab of services gewijzigd.

Het bestaande Git-publicatiepad blijft beschikbaar totdat de nieuwe route is
gecontroleerd. Alleen `WIND_PAGES_PUBLICATION_MODE=artifact` of de CLI-optie
`--pages-publication-mode artifact` activeert de nieuwe route. De bestaande
`--git-auto-push-pages` vlag schakelt publicatie in voor beide transports.

Bescherm altijd:

- de laatste current-day PNG en CSV per lokale dag, per canonieke site;
- alle bestanden in `rider_activity_uploads` en `rider_activity_analysis`, ook
  SVG, HTML, kaarten en `runs.csv`; ook oude rideranalyses zijn geen algemene cache;
- actuele website-PNG's, CSV-downloads, interactieve JSON, HTML, JS en iconen;
- de canonical database, het ECMWF-archief, modellen/scalers en model-gate-evaluaties.

Een plot reconstrueren met de huidige code levert niet noodzakelijk exact het
oude beeld op. De oorspronkelijke laatste dagplot blijft daarom beschermd
tijdens de migratie; controleer een reconstructie voordat die iets vervangt.

## Gewenste opslagstructuur

```text
next_day_wind_model/artifacts/<site>/
  current_day_predictions.csv             # actueel; vaste naam
  current_day_predictions.png             # actueel; vaste naam
  current_day_predictions_mobile.png      # actueel; vaste naam
  current_day_plot_archive/
    current_day_predictions_YYYY-MM-DD.png # blijvend: laatste render van die dag
  current_day_csv_archive/
    current_day_predictions_YYYY-MM-DD.csv # blijvend: inclusief gerealiseerde wind
  intraday_prediction_cache/YYYYMMDD/
    <uur_en_UTC_offset>_target_YYYYMMDD.csv # tijdelijk: eerste voorspelling per uur
    latest.csv                            # tijdelijk: laatste voorspelling
  dayahead_snapshots/
    dayahead_target_YYYYMMDD.csv           # eerste issue; evaluatiefallback

next_day_wind_model/publication/dashboard.zip # één vervangbaar publicatiepakket
```

De huidige site is `valkenburgsemeer`. Dezelfde functies werken met een eigen
artifactmap voor andere geregistreerde spots. Het portal gebruikt de spot van
de rit bij het kiezen van het archief en behoudt de oude archief-URL's.

Alle drie renderpaden (volledige voorspelling, cached-dashboard, alleen nieuwe
metingen) schrijven het dagpaar bij. De datum komt uit de CSV, zodat een
middernachtrun de vorige dag niet verkeerd labelt. Een tabel met minder
waarnemingen mag een vollediger dagpaar niet vervangen. Mobiele dagarchieven
die al bestaan blijven behouden; nieuwe permanente exports zijn één desktopplot
en één CSV per dag. De beschermde dagplot toont de volledige lokale kalenderdag,
zodat nachtelijke waarnemingen buiten de gewone websiteweergave 08:00–22:00
ook zichtbaar zijn. De websiteweergave blijft ongewijzigd. Next-day
PNG-archivering stopt.

De grafiek gebruikt overdag de eerste voorspelling van ieder uur voor de
historische lijnen. Daarom blijft een begrensde uurcache bestaan. Zodra een
dag is gesloten en een PNG/CSV-dagpaar de waarnemingen afdekt, verwijdert de code
uitsluitend zijn eigen herkenbare bestanden uit die tijdelijke cache.
Legacy `current_day_snapshots` worden nooit automatisch verwijderd.

De vroegste day-ahead CSV per doeldag blijft voorlopig als uitzondering behouden:
de historische MAE-code heeft een fallback naar die eerste uitgifte. Een laatste
current-day CSV heeft een andere betekenis. Schrap deze uitzondering pas nadat
de dekking van `prediction_log` voor alle historische dagen is aangetoond en de
fallback is vervangen. Nieuwe runs maken hiervan maximaal één bestand per doeldag.

De alleen-lezen controle op 16 oktober bevestigde dat de canonical intraday- en
next-day-logs pas op 5 april 2026 beginnen. Er bestaat daarnaast een aparte
next-day-backfill tot begin april. Neem daarom niet aan dat iedere oude CSV
dezelfde voorspelling volledig in de database heeft. Bewaar vóór CSV-opruiming
ook een externe kopie van de historische snapshots; controleer oudere perioden
afzonderlijk. Beschikbare canonical waarnemingen alleen bewijzen niet dat alle
destijds uitgegeven voorspellingen zijn vastgelegd.

## 1. Inventariseren zonder verwijderen

Gebruik de ontwikkelinterpreter vanuit de ontwikkelwerkmap:

```bash
python scripts/plan_runtime_cleanup.py \
  --artifact-root /pad/naar/productie/next_day_wind_model/artifacts \
  --artifact-root /pad/naar/productie/next_day_wind_model/artifacts/valkenburgsemeer \
  --artifact-root /pad/naar/productie/next_day_wind_model/artifacts/oostvoorne \
  --output storage_cleanup_reports/runtime-plan.json
```

Het rapport bevat kandidaatbestanden, beschermde bestanden en geblokkeerde
CSV's, inclusief grootte, inode en wijzigingstijd. Het script heeft bewust geen
verwijderoptie. Onbekende namen, symlinks, actieve dagen en de laatste
timestampplot per dag/type blijven beschermd. Het doorzoekt geen riderdata of
live webdirectory. De bestaande `plan_png_archive_cleanup.py` valt buiten deze
procedure: dat oudere script kent de nieuwe CSV- en sitecontroles niet.

Meting op 16 oktober 2026, productielegacy + Valkenburg:

| Classificatie | Bestanden | Bytes | Betekenis |
| --- | ---: | ---: | --- |
| Kandidaat | 4.664 | 794.651.731 | Oudere current-day PNG's, reconstructible next-day PNG's en latere day-ahead CSV's |
| Beschermd binnen de gescande archieven | 1.013 | 244.281.590 | Onder andere de laatste timestampplots en permanente dagplots |
| Geblokkeerd | 42.997 | 2.148.570.823 | Eerst een gecontroleerde dag-CSV met waarnemingen behouden |

Dit zijn runtimebestanden buiten Git. Git-objectopslag wordt hierdoor niet
kleiner. Werkmappen delen één `.git`; tel ze niet als zelfstandige Git-kopieën.

## 2. Dagparen eerst reconstrueren en controleren

Maak voor iedere historische site/dag eerst een stagingexport. Voorbeeld:

```bash
python scripts/finalize_daily_archive.py \
  --source-artifacts /pad/naar/productie/next_day_wind_model/artifacts/valkenburgsemeer \
  --db /pad/naar/canonical/wind_data_all_sites.db \
  --site valkenburgsemeer --day 2026-10-15 \
  --output-dir storage_cleanup_reports/staged-daily
```

De tool opent de database alleen-lezen, combineert een bestaande forecasttabel
met alle canonical waarnemingen van die lokale dag en gebruikt de bestaande
plotrenderer. Er worden geen voorspellingen getraind of gelogd en geen
migraties uitgevoerd. Het verificatierapport bevat het aantal waarnemingen,
de laatste waarnemingstijd en SHA-256 van PNG en CSV. Het doel moet geïsoleerd
zijn en bestaande dagexports worden niet overschreven.

Controleer de CSV en afbeelding, inclusief het laatste beschikbare meetmoment,
en vergelijk met de behouden oorspronkelijke laatste plot. Ontbreken brondata
of forecastkolommen, dan stopt de tool en blijven de legacybestanden behouden.
Bewaar ook dagen die uitsluitend in het legacyarchief staan; kopieer nooit
blind alleen de nieuwe sitemap. Neem nieuwe sitearchieven en alle riderdata
op in de externe back-up; een SQLite-back-up dekt die bestanden niet.

Promoveer gecontroleerde dagparen pas met de updaterlock vast en met een
geverifieerde back-up. Als de oorspronkelijke laatste plot volledig is, kan
die behouden worden in plaats van de gereconstrueerde PNG. Controleer daarna
de ritpagina en de oude plus site-specifieke archief-URL. Verwijder nooit een
oorspronkelijke laatste plot alleen omdat een gelijknamige dagplot bestaat.

## 3. Publicatie omzetten, bestaande dashboard-URL behouden

Nieuwe keten:

```text
updater onder bestaande lock
  -> actuele websitebestanden
  -> gecontroleerd dashboard.zip, atomair vervangen
  -> publieke portalroute /forecast-publication/dashboard.zip
  -> GitHub Actions workflow_dispatch
  -> download + manifest/hash/linkcontrole
  -> upload-pages-artifact (retentie 1 dag)
  -> deploy-pages naar dezelfde Pages-site
```

Er zijn geen Git add/commit/push-stappen in artifactmodus. De allowlist omvat
alle publieke huidige dashboardbestanden, inclusief CSV-downloads, D+2,
evaluatiepagina's en iconen, en registreert gepubliceerde extra spots.
Riderbestanden, databasebestanden, dagarchieven en operationele documenten
gaan niet mee. Een foutief pakket vervangt het vorige lokale pakket niet;
een mislukte workflow-download/validatie komt niet bij de deploymentstap.
Een dispatchbevestiging betekent 'aangevraagd', niet 'succesvol gedeployed'.

Eenmalige migratie, met terugvalplan:

1. Leg de productiecommit, crontab en huidige Pages-bron vast. Maak en verifieer
   de vereiste back-ups voordat productiecode of archieven worden gewijzigd.
2. Neem de geteste code en `deploy-dashboard.yml` op in de productietak, met
   de Git-transportroute nog actief. Herstart het portal pas in het afgesproken
   deploymentvenster. De productie-updater is nooit een testcommando.
3. Bouw met de nieuwe code een pakket van de succesvolle publicatie onder de
   bestaande updaterlock. De standaardlocatie is de genegeerde directory
   `next_day_wind_model/publication/`. Met `WIND_DASHBOARD_BUNDLE_PATH` kan dit
   naar externe runtimeopslag; stel dezelfde absolute waarde in voor cron en portal.
4. Controleer vanaf buiten de server de HTTPS-bundleroute en voer `extract`
   lokaal uit. HTTPS-bereikbaarheid door GitHub-hosted runners is een verplichte
   canary; een lokale test vervangt deze niet.
5. Configureer de repositoryvariabele `WIND_DASHBOARD_BUNDLE_URL` met de exacte
   HTTPS-URL van deze portalroute. Configureer `WIND_GITHUB_ACTIONS_TOKEN` in een
   beveiligde omgeving voor de cron-gebruiker, met repositorytoegang en
   Actions-write voor `workflow_dispatch`. De publisher gebruikt de GitHub API
   rechtstreeks en vereist geen extra CLI-installatie. Zet credentials
   nooit in cron, Git, uitvoer of het zipbestand. Controleer actionstoestemming,
   environmentbescherming en het runnerbudget van deze repository.
6. Stel de Pages-bron in op **GitHub Actions** en start één gecontroleerde
   workflow. Controleer de workflow én de bestaande publieke URL: current/next
   day, D+2, mobiel, evaluaties, CSV's, iconen en tijdstempels.
7. Zet pas na deze canary `WIND_PAGES_PUBLICATION_MODE=artifact` in de actieve
   cron-omgeving. Zowel de wrapper als de dagelijkse trainingsjob lezen dit.
   Controleer twee normale zesminutenruns én de volgende kalenderdagovergang.
8. Bevestig dat de bron-HEAD en objecttelling niet meer groeien door dashboard-
   verversingen en dat Pages werkelijk nieuwere metingen toont.

Totdat stap 6 geslaagd is, blijven de gegenereerde `docs`-bestanden gevolgd in
Git. Haal ze daarna alleen uit de Git-index; laat ze fysiek staan als invoer
voor het pakket. Selecteer uitsluitend bekende publieke paden:

```python
import subprocess
from next_day_wind_model.dashboard_publication import allowed_files
tracked = subprocess.check_output(["git", "ls-files", "-z", "--", "docs"]).decode().split("\0")
paths = [p for p in tracked if p.startswith("docs/") and p[5:] in allowed_files()]
# Alleen uitvoeren na een geslaagde externe Pages-canary en gecontroleerde diff:
subprocess.run(["git", "rm", "--cached", "--", *paths], check=True)
```

De `.gitignore`-regels verhinderen vervolgens dat runtimebestanden per ongeluk
opnieuw worden opgenomen. Broniconen/templates onder `web_assets` blijven in Git.

Terugval: zet het transport terug op `git`, herstel de oude Pages-bron en
publiceer de behouden actuele websitebestanden via de bestaande route. Pas de
archieven of database niet aan om publicatie terug te draaien. Doe dit vóór
een eventuele history rewrite, zodat de oude route beschikbaar blijft.

Deze route gebruikt de officiële Pages-artifactworkflow:
[GitHub Pages custom workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).
Houd rekening met de repositoryspecifieke Actions-opslag en uitvoertijd; de
publieke site blijft dezelfde URL houden.

## 4. Runtimebestanden gecontroleerd opruimen

Na de geslaagde dagarchiefmigratie en portalcontrole:

1. Maak een nieuw read-only plan; controleer dat beschermde dagparen voor alle
   historische dagen en spots aanwezig zijn. Eerdere rapporten zijn geen
   toestemming om ondertussen gewijzigde bestanden weg te halen.
2. Verifieer een externe kopie van beschermde dagparen, alle riderbestanden en
   de bronbestanden die voorlopig in quarantaine gaan. Voldoende ruimte voor
   eventuele reconstructie moet op die andere opslag beschikbaar zijn.
3. Neem de updaterlock en controleer per kandidaat opnieuw pad, inode, grootte,
   wijzigingstijd, dag en beschermde vervanger. Verplaats uitsluitend de
   gecontroleerde manifestselectie naar quarantaine. Gebruik geen brede
   `find -delete`, `git clean`, `rm *.csv` of zoekactie over riderdata.
4. Test ritpagina's en dagplot-URL's van vroege, recente en migratiedagen en
   verschillende spots; controleer numerieke voorspellinglijnen en evaluaties.
5. Verwijder de quarantaine pas na de afgesproken herstelperiode en expliciete
   vrijgave. Quarantaine op hetzelfde filesystem levert nog geen vrije ruimte;
   verplaats naar geverifieerde andere opslag om direct ruimte terug te krijgen.

Voeg legacy current-day CSV's pas toe als de dag-CSV de waarnemingen afdekt;
de planner blokkeert ze anders. Behoud de vroege day-ahead-evaluatiefallback en
model-gate-details totdat hun afzonderlijke afhankelijkheid is gemigreerd.

## 5. Git-geschiedenis apart verkleinen

Pas uitvoeren nadat nieuwe verversingen geen gegenereerde commits meer maken.
De oorspronkelijke analyse vond circa 20,7 GiB aan PNG-objecten in Git. Een
runtimeopruiming of `git rm --cached` haalt die oude objecten niet weg.

Maak op andere opslag een volledige, gecontroleerde kopie van de repository,
alle refs/stashes en de beschermde dagarchieven. Filter in een aparte kopie
uitsluitend de bekende gegenereerde publicatiepaden uit alle relevante refs;
behoud broncode, documentatie, templates en broniconen. Behandel lokale branches,
remote refs, reflogs en worktrees als deel van dezelfde migratie. Controleer een
schone clone, de site en herstelbaarheid voordat de remote history of
productiecheckouts worden vervangen. Een terugvalkopie op het rootfilesystem
zou de beoogde ruimtewinst tenietdoen. Geen aggressive GC of handmatig
verwijderen van packfiles op de huidige bijna volle productieschijf.

## Validatie

```bash
python -m pytest tests next_day_wind_model -q
python -m next_day_wind_model.dashboard_publication build \
  --web-root docs --bundle storage_cleanup_reports/dashboard.zip
python -m next_day_wind_model.dashboard_publication extract \
  --bundle storage_cleanup_reports/dashboard.zip --destination storage_cleanup_reports/publication-preview
```

Tests omvatten uurcache, zomertijd, middernacht, laatste waarnemingen,
CSV-bescherming, oude en nieuwe portal-URL's, publicatiebestanden, ontbrekende
assets, symlinks, ZIP-padtraversal/checksums, mislukte dispatch en herhaald
publiceren zonder nieuwe Git-objecten. Een echte GitHub Pages-canary blijft
onderdeel van de productionele omschakeling.

Uitgevoerd op 16 oktober 2026: **323 tests en 74 subtests geslaagd**. De 42 echte
websitebestanden zijn na verpakken en uitpakken byte voor byte gelijk. Een
lokale Chromium-controle is uitgevoerd op desktop, mobiel en een fictieve
portalrit; de site-specifieke archiefplot laadt met HTTP 200. Voor 15 oktober
bevat de staging-CSV alle 478 canonical waarnemingen, tot 23:59:43 lokale tijd.
De volledige dagplot is visueel gecontroleerd. Er is geen echte workflow naar
GitHub verstuurd; externe HTTPS-bereikbaarheid, Actions-authenticatie en de
live Pages-publicatie moeten tijdens de canary worden bevestigd.
