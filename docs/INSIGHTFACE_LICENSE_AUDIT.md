# InsightFace License Audit (FaceTrak 2.1)

Stand: 2026-10-16. Geprüft gegen die live-Repositories und Herstellerangaben
(siehe Quellen unten). Code-Lizenzen und Modellgewichte werden **separat**
bewertet — das ist der entscheidende Punkt der InsightFace-Lizenzpolitik.

## Ergebnis im Überblick

| Komponente | Code-Lizenz | Gewichte-Lizenz | In FaceTrak |
|---|---|---|---|
| InsightFace Python-Paket / App-Framework | MIT | — (Code) | nur Konzept-Referenz, kein Vendor-Code kopiert |
| SCRFD-Detektor-Code (OpenMMLab-Basis) | Apache-2.0 | — (Code) | eigene Implementierung (`recog/scrfd.py`), kein Copy-Paste |
| `buffalo_*` / `antelopev2` / `raccoon_*` Gewichte | — | **nicht-kommerziell, Forschung only** | **nicht gebündelt, kein Auto-Download** |
| ArcFace `w600k_*` / `glintr100` Gewichte | — | **nicht-kommerziell, Forschung only** | **nicht gebündelt, kein Auto-Download** |
| GenderAge / Landmark-Modelle (InsightFace) | — | **nicht-kommerziell, Forschung only** | **nicht gebündelt** |
| PrivateFrame-Workflow | MIT | — | nur Architekturkonzept übernommen, eigene Implementierung |
| ONNX Zoo `arcfaceresnet100` (Intel/Amazon) | Apache-2.0 (Model Card) | Apache-2.0 laut Model Card | nicht gebündelt; Hinweis: Trainingsdaten-Herkunft (MS-Celeb-1M, zurückgezogen) separat prüfen |
| YuNet / SFace (OpenCV Zoo, Bestand) | Apache-2.0 | Apache-2.0 | unverändert gebündelt wie bisher |
| ONNX Runtime (optional) | MIT | — | optionale Dependency `insightface`-Extra |

**Kommerzielle Nutzung von InsightFace-Modellgewichten erfordert eine
Lizenz von InsightFace** (insightface.ai → Licensing). FaceTrak verteilt
keine solchen Gewichte und lädt keine automatisch herunter.

## Durchsetzung im Code (Lizenz-Gates)

1. `insightface_adapter.py`: `allow_auto_download=False` Default; keine
   Download-Funktion für InsightFace-Gewichte vorhanden. Fehlendes
   Modellverzeichnis → Adapter meldet `available=False` + `init_error`,
   Engine fällt auf YuNet/SFace zurück.
2. `device/management.py` `ModelRegistry`: `auto_download=False`,
   `require_license_consent=True`. `POST /api/v1/model-registry/register`
   verweigert kommerziell lizenzpflichtige Modelle ohne
   `license_consent=true` mit HTTP 403.
3. Backend-Switch (`POST /api/v1/face/backends`) verweigert `scrfd`/`arcface`
   mit HTTP 409, solange kein lizenziertes Modell geladen ist.
4. `models/`-Verzeichnis ist kein Repo-Inhalt (nur lokale Ablage);
   es werden keine `.onnx`-Gewichte von InsightFace committet.

## Quellen

- `deepinsight/insightface` README: „The code … is released under the MIT
  License … The training data … (and the models trained with these data)
  are available for non-commercial research purposes only.“
- `deepinsight/insightface` `server/LICENSING.md`: Server-Code MIT;
  Open-Source-Modellpakete nur nicht-kommerzielle Forschung;
  jedes Modellverzeichnis führt eine `MODEL.LICENSE`-Datei.
- `insightface.ai` Licensing-Seiten: kommerzielle Lizenzen für
  `buffalo_l`, `antelopev2`, `buffalo_s`, `buffalo_m` etc. erforderlich.
- `detection/scrfd/LICENSE` (im InsightFace-Repo): OpenMMLab Apache-2.0.
- ONNX Model Zoo `arcfaceresnet100`: Apache-2.0 Model Card; Paper
  ArcFace (CVPR 2019), Trainingsdaten MS-Celeb-1M (von Microsoft
  zurückgezogen — eigene Prüfung bei Nutzung dieser Gewichte).
- `python-package/insightface/app/privateframe/README.md`: Workflow
  (analysieren → JSON editieren → rendern ohne Re-Inferenz), MIT-Code.

## Offene Pflichten des Betreibers

- InsightFace-Gewichte nur mit gültiger kommerzieller Lizenz oder
  rein nicht-kommerzieller Forschung einsetzen.
- `MODEL.LICENSE`-Datei je Modellverzeichnis mitführen.
- Referenzfotos für selektive Anonymisierung nur mit Einwilligung.
- Keine verdeckte Identifizierung: Anonymisierung läuft standardmäßig
  ohne Recognition (`recognition_backend=none`); keine Auto-Registrierung,
  keine kameraübergreifende Re-ID (FaceTracker-ReID nur innerhalb
  8 s nach Track-Verlust, keine Persistenz).
