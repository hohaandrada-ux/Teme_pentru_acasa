# Tema 3 – Testare si Evaluare Model AI

## Descriere
In cadrul acestei teme am implementat teste unitare si metrici de evaluare pentru un model simplu de clasificare a sentimentului.

## Testare
Am utilizat stilul pytest pentru a valida comportamentul modelului prin teste unitare.

Au fost acoperite urmatoarele cazuri:
- clasificare pozitiva, negativa si neutra
- input gol
- case insensitivity (litere mari/mici)
- situatii ambigue (ex: "Nu imi place")

## Metrici de evaluare
Am implementat doua metrici:
- **Accuracy** – masoara proportia predictiilor corecte
- **Precision (negativ)** – masoara cat de corecte sunt predictiile pentru clasa negativa

## Evaluare
Modelul este evaluat pe un set de exemple, iar rezultatele sunt afisate impreuna cu predictiile si valorile asteptate.

## Observatii
Modelul este unul simplificat, bazat pe reguli simple, si a fost folosit pentru a demonstra procesul de testare si evaluare conform conceptelor prezentate in lectie.
