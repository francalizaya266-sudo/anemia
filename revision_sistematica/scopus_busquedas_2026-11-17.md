# Búsqueda de citas en Scopus (observación 4 del docente)

Fecha: 17 de noviembre de 2026, entre las 16:42 y las 17:02 UTC (11:42 a 12:02, hora de Lima). Acceso: suscripción institucional de la universidad. Los registros se leyeron de las páginas de resultados; no se exportó ni se descargó ningún archivo.

Archivos:

- `scopus_citantes_cp-anemic_2026-11-17.csv`: 37 registros. Los 32 documentos que Scopus lista como citantes del artículo de datos y 5 que solo salen en las búsquedas por texto. La columna `en_wos` dice si ya estaban en la búsqueda de Web of Science y la columna `segun_resumen` dice qué datos usa cada uno.
- `scopus_tema_conjuntiva_anemia_2026-11-17.csv`: 119 registros de la búsqueda temática (búsqueda 7), con el identificador de Scopus de cada uno. La lista de resultados no muestra el DOI.
- `extraccion_busqueda_citas_2026-11-17.csv`: tabla de extracción. La fila S16 (Berghout 2024) se completó y hay tres filas nuevas (S17 a S19), leídas solo por su resumen.

## Búsquedas

| # | Cadena exacta | Hora UTC | n |
|---|---|---|---|
| 1 | `DOI(10.1016/j.medntd.2023.100244)` | 16:42 | 1 (Cited by 32) |
| 2 | Lista «Cited by» del registro 2-s2.0-85161610659; Scopus la ejecuta como `REF(2-s2.0-85161610659)` | 16:43 | 32 |
| 3 | `TITLE-ABS-KEY("CP-AnemiC" OR "CP-Anemic" OR "CPAnemiC")` | 16:53 | 6 |
| 4 | `REF("CP-AnemiC")` | 16:54 | 36 |
| 5 | `REF("m53vz6b7fx") OR REF("nt7r8hv2pz")` | 16:55 | 0 |
| 5b | `REF("Mendeley Data" AND "Conjunctival Pallor")` | 16:56 | 4 |
| 6 | `ALL("CP-AnemiC")` | 16:56 | 38 |
| 7 | `TITLE-ABS-KEY(conjunctiva* AND (anemia OR anaemia) AND ("deep learning" OR "machine learning" OR "convolutional")) AND PUBYEAR > 2018 AND PUBYEAR < 2027` | 16:57 | 119 |
| 8 | Comprobación por DOI de 19 registros de Web of Science (19 términos `DOI(...)` unidos con OR) | 17:02 | 15 |

Registro del artículo de datos en Scopus: EID 2-s2.0-85161610659.

## Lo que hay que saber antes de usar los números

- **32 citantes en Scopus frente a 25 declarados (18 visibles) en Web of Science.** 18 documentos están en las dos listas y 14 solo en la de Scopus.
- **Cinco documentos más salen solo en las búsquedas por texto** (3, 4, 5b y 6), porque Scopus no los enlaza al registro: Romero y Beauchamp 2026, Bhoye y Mhala 2026, Sindhusha et al. 2025 (ICIMIA), Berghout 2024 (CCE) y Yakimov et al. 2024 (Scientific Reports). El último ya estaba en Web of Science.
- **La búsqueda 5 no sirve.** `REF` no encuentra el DOI de Mendeley Data dentro de la referencia. La 5b busca el texto de la referencia y da 4.
- **Cuatro registros de Web of Science no están en Scopus** (comprobado por DOI): Parikh et al. 2024, Uvaliyeva et al. 2024, Mahendra et al. 2026 y Sumihar et al. 2025. Los dos últimos son los estudios que la búsqueda en Web of Science añadió a la Tabla 3: una búsqueda hecha solo en Scopus no los habría encontrado.
- Denisenko et al. 2026 está en Scopus, pero ninguna búsqueda por referencia lo devuelve. No se buscó el preprint de arXiv de Cruz Romero y Lugo Beauchamp; sí está su versión publicada (Romero y Beauchamp 2026).
- **Búsqueda temática:** 119 registros. 16 citan o mencionan CP-AnemiC, 7 son actas completas de congresos y 1 es el propio artículo de datos. En Web of Science la misma búsqueda dio 47.
- No se revisaron las pestañas «Preprints» ni «Secondary documents».
- Dos registros no abrieron por un error del servidor de Scopus (Bhoye y Mhala 2026; Asare et al. 2024); sus resúmenes se leyeron en la lista de resultados.
- Scopus guarda por sí solo las 12 consultas en el historial de búsquedas de la sesión. No se guardaron búsquedas ni se crearon alertas o listas.

## Qué hace cada uno de los 37 registros

Lo que sigue sale del resumen y las palabras clave del registro en Scopus, salvo donde dice «texto completo leído».

Usan CP-AnemiC para clasificar la anemia, texto completo leído (3):

- Fuentes-Beingolea et al. 2025, iJOE. Ya estaba en la Tabla 3.
- Romero y Beauchamp 2026, Lecture Notes in Networks and Systems. Es la versión publicada del preprint de Cruz Romero y Lugo Beauchamp 2025, que ya estaba en la Tabla 3. Se leyó el preprint, no la versión publicada, que es de pago.
- Berghout 2024, Journal of Imaging. **Nuevo en la Tabla 3.** Leído de forma dirigida en PubMed Central (PMC11508579). Describe 710 fotos de conjuntiva (424 y 286) y cita los registros de Mendeley Data; no nombra CP-AnemiC en el cuerpo. Indica que los autores de los conjuntos los distribuyeron con imágenes aumentadas (rotaciones de 90° y 270°, volteos y pequeños desplazamientos) y que las descartó. Equilibra las clases con datos sintéticos y reporta 99.65 % (LSTM) y 100 % (RexNet) en la conjuntiva. No dice cuántas imágenes usó al final, las proporciones de la partición ni si equilibró antes de partir.

Nombran CP-AnemiC en el resumen (3):

- Paul et al. 2026, QPAIN 2026. Evalúa en Eyes-Defy y en CP-AnemiC con rasgos de color y textura y clasificadores clásicos.
- Rooj, Vyas y Bharti 2026, CCIS 2685. SE-MobileNet.
- Sindhusha et al. 2025, ICIMIA 2025. Habla de 2000 imágenes de conjuntiva; reporta 96.8 % de exactitud.

Usan imágenes de conjuntiva de un conjunto que el resumen no nombra, o no tienen resumen (6):

- Asare et al. 2026, CCIS 2723 (CNN híbrida de tres capas). Su lista de referencias cita el artículo de datos y el registro de CP-AnemiC en Mendeley Data (10.17632/m53vz6b7fx.1).
- Asare et al. 2024, Informatics in Medicine Unlocked. Niños; partición aleatoria 70:10:20; CNN con 98.45 % de exactitud.
- Berghout 2024, CCE 2024. Gravedad de la anemia en cuatro clases. Cita el conjunto de Mendeley Data.
- Chakrabarty et al. 2025, UPCON 2025. «Un conjunto en línea»; YOLOv8; 99 % de exactitud.
- Atmakuri et al. 2025, GIEST 2025. Conjunto de Kaggle; exactitud 96.5 % y AUC 0.987.
- Esfahani et al. 2026, American Journal of Hematology. Carta sin resumen.

Usa CP-AnemiC para otra cosa, texto completo leído (1): Alhjori et al. 2026, Diagnostics.

No usan CP-AnemiC, según el texto leído o el resumen (24):

- Otros datos de conjuntiva: Sindhusha et al. 2026 (Eyes-defy-anemia), Lee Huamaní et al. 2026 (202 imágenes de otro conjunto; ya citado en el manuscrito), Camporeale et al. 2025 (datos propios), Fuentes-Beingolea y Palomino-Quispe 2025 (conjunto público de 218 imágenes), Çuvadar y Yılmaz 2023 (388 donantes).
- Palma o uñas: Asare et al. 2026 (CCIS, palma y uñas), Ramzan et al. 2024 (VCIBA), Appiahene et al. 2023, Ccaso et al. 2024, Krithika et al. 2024.
- Sin imágenes: Asare et al. 2026 (prevalencia), Said y Tunga 2025 (datos tabulares de Tanzania), Bhoye y Mhala 2026.
- Revisiones: Rooj et al. 2026, Ersianti et al. 2026.
- Métodos ópticos con datos propios: Yakimov et al. 2024 (Scientific Data), Yakimov et al. 2024 (Scientific Reports), Bardadin et al. 2024.
- Sin relación con la anemia: Asare et al. 2026 (infección urinaria), Asare et al. 2026 (viruela del mono), Asare et al. 2025 (melanoma), Asare et al. 2025 (rabia), Sharma y Guleria 2025 (neumonía), Chintalapati et al. 2023 (préstamos).

## Qué cambia respecto de la lista de Web of Science

De los siete trabajos que quedaron sin leer en `wos_busquedas_2026-11-17.md`:

- Berghout 2024 (Journal of Imaging) ya se leyó y está en la Tabla 3.
- Cuatro no usan CP-AnemiC según su resumen en Scopus: Said y Tunga 2025, Sharma y Guleria 2025, Çuvadar y Yılmaz 2023, y Fuentes-Beingolea y Palomino-Quispe 2025.
- Parikh et al. 2024 no está en Scopus. Según su ficha en Semantic Scholar, propone un modelo LightGBM con marcadores clínicos y bioquímicos, sin imágenes. No se encontró el texto completo.
- Esfahani et al. 2026 sigue sin leer.

## Intento de leer los textos completos (17 de noviembre de 2026)

Se intentó abrir en el navegador el texto completo de 14 documentos, un intento cada uno, sin descargar archivos ni iniciar sesión. No se resolvió ninguna comprobación de robot.

| Documento | Resultado |
|---|---|
| Berghout 2024, Journal of Imaging | Leído en PubMed Central (la página de MDPI solo mostró el resumen) |
| Romero y Beauchamp 2026 | El capítulo de Springer es de pago; se leyó su versión de arXiv (2507.15151) |
| Asare et al. 2024, Informatics in Medicine Unlocked | No se pudo: comprobación de robot de ScienceDirect |
| Asare et al. 2023, Medicine in Novel Technology and Devices 18:100237 | No se pudo: comprobación de robot de ScienceDirect |
| Asare et al. 2023, Engineering Reports 5(11):e12667 | No se pudo: verificación de seguridad de Wiley |
| Esfahani et al. 2026, American Journal of Hematology | No se pudo: verificación de seguridad de Wiley |
| Parikh et al. 2024, INTI Journal | No se encontró el texto completo |
| Paul et al. 2026; Sindhusha et al. 2025 (ICIMIA); Berghout 2024 (CCE); Chakrabarty et al. 2025; Atmakuri et al. 2025 | No se pudo: IEEE Xplore es de pago |
| Rooj, Vyas y Bharti 2026; Asare et al. 2026 (CNN híbrida) | No se pudo: Springer es de pago |

Del preprint de Cruz Romero y Lugo Beauchamp se confirmó que el F1 de 0.977 del resumen es el mejor F1 de validación durante el entrenamiento (Tabla III) y que el F1 de inferencia es 0.943 (Tabla IV).

## Google Scholar

A las 17:12 UTC se buscó el título exacto del artículo de datos entre comillas. Google respondió con la página de «tráfico inusual» (HTTP 429), sin resultados. No se resolvió la comprobación y no se obtuvo el número de «Citado por» ni la lista. **El equipo decidió el 17-11-2026 no usar Google Scholar.** La búsqueda de citas del artículo se limita a Web of Science y Scopus.

## Lo que falta

- Leer el texto completo de once trabajos que podrían usar el conjunto y, si lo usan para clasificar, añadirlos a la Tabla 3:
  1. Paul et al. 2026 (QPAIN 2026), doi:10.1109/QPAIN69676.2026.11546379. Nombra CP-AnemiC.
  2. Rooj, Vyas y Bharti 2026 (CCIS 2685), doi:10.1007/978-3-032-08511-5_14. Nombra CP-AnemiC.
  3. Sindhusha et al. 2025 (ICIMIA 2025), doi:10.1109/ICIMIA67127.2025.11200595. Nombra CP-AnemiC.
  4. Asare et al. 2026 (CCIS 2723, CNN híbrida), doi:10.1007/978-3-032-13056-3_12.
  5. Asare et al. 2024 (Informatics in Medicine Unlocked 45:101451), doi:10.1016/j.imu.2024.101451. Acceso abierto.
  6. Berghout 2024 (CCE 2024), doi:10.1109/CCE62852.2024.10770882.
  7. Chakrabarty et al. 2025 (UPCON 2025), doi:10.1109/UPCON66414.2025.11454042.
  8. Atmakuri et al. 2025 (GIEST 2025), doi:10.1109/GIEST66547.2025.11387069.
  9. Esfahani et al. 2026 (American Journal of Hematology 101(2):361-364), doi:10.1002/ajh.70143.
  10. Asare et al. 2023, «Detection of anemia using conjunctiva images: A smartphone application approach» (Medicine in Novel Technology and Devices 18:100237), doi:10.1016/j.medntd.2023.100237. Acceso abierto. No cita el artículo de datos; salió en la búsqueda temática.
  11. Asare et al. 2023, «Iron deficiency anemia detection using machine learning models: A comparative study of fingernails, palm and conjunctiva of the eye images» (Engineering Reports 5(11):e12667), doi:10.1002/eng2.12667. Acceso abierto. No cita el artículo de datos; salió en la búsqueda temática.
- **Leer completo Berghout 2024 (Journal of Imaging), que es de acceso abierto.** Es la única mención publicada que encontramos de copias aumentadas en estos conjuntos. Hay que confirmar a qué conjuntos se refiere, cuántas imágenes de conjuntiva usó al final y cómo partió los datos.
- Confirmar contra el PDF las filas S16 a S19 de la tabla de extracción.
