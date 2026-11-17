# Búsqueda de citas en Web of Science (observación 4 del docente)

Fecha: 17 de noviembre de 2026, entre las 16:02 y las 16:18 UTC (11:02 a 11:18, hora de Lima). Acceso: suscripción institucional de la universidad. Los registros se leyeron de las páginas de resultados; no se exportó ni se descargó ningún archivo.

Archivos:

- `wos_citantes_cp-anemic_2026-11-17.csv`: 25 registros distintos que citan o nombran CP-AnemiC.
- `wos_tema_conjuntiva_anemia_2026-11-17.csv`: 47 registros de la búsqueda temática (búsqueda 11).
- `extraccion_busqueda_citas_2026-11-17.csv`: tabla de extracción, con cinco filas nuevas (S12 a S16).

## Búsquedas

| # | Cadena exacta | Base y ediciones | Hora UTC | n |
|---|---|---|---|---|
| 1 | `DO=(10.1016/j.medntd.2023.100244)` | Core Collection: SCI-EXPANDED, SSCI, AHCI (2011-presente), ESCI (2021-presente) | 16:02 | 1 (Times Cited 25) |
| 2 | Citantes de WOS:001454148800026 (enlace «25 Citations») | Core Collection, mismas ediciones | 16:04 | 18 visibles de 25 |
| 3 | `DO=(10.1016/j.medntd.2023.100244)` | All Databases | 16:08 | 1 (Times Cited 29) |
| 4 | Citantes de WOS:001454148800026 | All Databases | 16:09 | 21 visibles de 29 |
| 5 | `TS=("CP-AnemiC" OR "CP-Anemic" OR "CPAnemiC")` | Core Collection | 16:10 | 2 |
| 6 | `TS=("CP-AnemiC" OR "CP-Anemic" OR "CPAnemiC")` | All Databases | 16:11 | 5 |
| 7 | Cited Reference Search, Cited DOI = `10.17632/m53vz6b7fx*` | Core Collection | 16:13 | 1 referencia citada (2 citantes declarados, 1 visible) |
| 8 | Cited Reference Search, Cited DOI = `10.17632/nt7r8hv2pz*` | Core Collection | 16:14 | 0 |
| 9 | Cited Reference Search, Cited Author = `Appiahene P* OR Asare J*` AND Cited Year(s) = `2023` | Core Collection | 16:14 | 62 referencias citadas; 3 son de CP-AnemiC |
| 10 | `TS=(conjunctiva* AND an?emia AND ("deep learning" OR "machine learning" OR "convolutional")) AND PY=(2019-2026)` | Core Collection | 16:17 | 4 |
| 11 | `TS=(conjunctiva* AND an$emia AND ("deep learning" OR "machine learning" OR "convolutional")) AND PY=(2019-2026)` | Core Collection | 16:18 | 47 |

Registro del artículo de datos: Appiahene P; Chaturvedi K; Asare JW; Donkoh ET; Prasad M. Medicine in Novel Technology and Devices 18, art. 100244, junio de 2023. ESCI. WOS:001454148800026. 33 referencias.

## Lo que hay que saber antes de usar los números

- **25 citas declaradas, 18 visibles.** La suscripción cubre SCI-EXPANDED, SSCI y AHCI (2011-presente) y ESCI (2021-presente). Los otros 7 citantes están en ediciones no suscritas y no se pueden ver con esta cuenta.
- **All Databases: 29 citas declaradas, 21 visibles.** El desglose de Web of Science es 25 en Core Collection, 3 en BIOSIS Citation Index y 3 en Research Commons. Los 21 visibles son los 18 de Core Collection y 3 de Research Commons.
- **Un citante que Web of Science no enlaza.** La búsqueda 9 muestra una variante mal formada de la referencia (sin volumen ni DOI) con 1 citante: Yakimov et al., Scientific Reports 2024.
- **La búsqueda 10 se queda corta.** En Web of Science `?` exige exactamente un carácter, así que `an?emia` solo encuentra «anaemia». Con `an$emia` (cero o un carácter) salen 47. Se reportan las dos.
- Web of Science guarda por sí solo las 11 consultas en el historial de búsquedas de la cuenta.

## Qué hace cada uno de los 25 registros

Usan CP-AnemiC para clasificar la anemia, texto completo leído (4):

- Fuentes-Beingolea et al. 2025, iJOE. Ya estaba en la Tabla 3.
- Cruz Romero y Lugo Beauchamp 2025, preprint. Ya estaba en la Tabla 3.
- Mahendra et al. 2026, Journal of Information Systems and Informatics. **Nuevo en la Tabla 3.**
- Sumihar et al. 2025, TIN: Terapan Informatika Nusantara. **Nuevo en la Tabla 3.**

Usa CP-AnemiC para otra cosa, texto completo leído (1):

- Alhjori et al. 2026, Diagnostics. Lo usa como conjunto externo para comparar imágenes sintéticas con imágenes reales. Se menciona en la sección 4.3, fuera de la tabla.

Solo lo citan, texto leído (2):

- Sindhusha et al. 2026, Scientific Reports. Usa Eyes-defy-anemia.
- Ramzan et al. 2024, Visual Computing for Industry, Biomedicine, and Art. Usa imágenes de la palma (leído en la búsqueda web).

**Podrían usar el conjunto y no se pudo leer el texto completo (7). El equipo debe leerlos:**

1. Berghout 2024, Journal of Imaging, doi:10.3390/jimaging10100245. Lectura parcial: usa 710 imágenes de conjuntiva (424 y 286) junto con palma y uñas, y el resumen reporta 99.83 ± 0.02 %. No se pudieron leer la partición ni los resultados.
2. Fuentes-Beingolea y Palomino-Quispe 2025, IEEE Access, doi:10.1109/ACCESS.2025.3638640. Estimación de Hb con imágenes de conjuntiva.
3. Esfahani et al. 2026, American Journal of Hematology (carta), doi:10.1002/ajh.70143.
4. Said y Tunga 2025, Machine Learning with Applications, doi:10.1016/j.mlwa.2025.100741.
5. Sharma y Guleria 2025, Journal of Transformative Technologies and Sustainable Development, doi:10.1007/s41314-025-00082-5.
6. Parikh et al. 2024, INTI Journal. El DOI que da Web of Science (10.61453/jods.v2024no39) abre otro artículo; hay que buscarlo por título.
7. Çuvadar y Yilmaz 2023, Medical Engineering & Physics, doi:10.1016/j.medengphy.2023.104038. Por la fecha, probablemente usa datos propios.

No se leyeron y, por el título, no usan las imágenes de conjuntiva de CP-AnemiC (11): dos revisiones (Rooj et al. 2026; Ersianti et al. 2026), cuatro trabajos del grupo que publicó el conjunto sobre otros temas (Asare et al. 2026, prevalencia; Asare et al. 2026, infección urinaria; Asare et al. 2025, melanoma; Appiahene et al. 2023, palma), cuatro trabajos de métodos ópticos con datos propios (Denisenko et al. 2026; Yakimov et al. 2024, Scientific Data; Yakimov et al. 2024, Scientific Reports; Bardadin et al. 2024) y Uvaliyeva et al. 2024.

## Posible trabajo relacionado, sin leer

- Freitas et al. 2026, «Cross-validation strategies under data dependency: An example with anemia prediction in sheep using ocular conjunctiva images», Preventive Veterinary Medicine 251 (WOS:001714508900001). Trata la dependencia entre datos en la validación cruzada, en imágenes de conjuntiva de ovejas. Si se lee, podría citarse en la sección 4.3.

## Actualización tras la búsqueda en Scopus (mismo día)

La búsqueda en Scopus está en `scopus_busquedas_2026-11-17.md`. Cambia esta lista así: Berghout 2024 ya se leyó y entró en la Tabla 3; Said y Tunga 2025, Sharma y Guleria 2025, Çuvadar y Yılmaz 2023, y Fuentes-Beingolea y Palomino-Quispe 2025 no usan CP-AnemiC según su resumen en Scopus; Parikh et al. 2024 no usa imágenes según su ficha en Semantic Scholar; Esfahani et al. 2026 sigue sin leer. La lista vigente de trabajos por leer (once) está en el archivo de Scopus.

## Lo que falta

- La búsqueda de citas queda limitada a Web of Science y Scopus. El equipo decidió el 17-11-2026 no usar Google Scholar.
- Leer los once trabajos listados en `scopus_busquedas_2026-11-17.md` y, si usan CP-AnemiC para clasificar, añadirlos a la Tabla 3.
- Confirmar contra el PDF las filas nuevas de la tabla de extracción (S12 a S16): se leyeron por extracción automática.
