# Resultado de la calificación a ciegas de los pares de la etapa 4 (kit v2)

Calificado el 17-11-2026 con `codigo/review7/kappa_inspection_v2.py`. Las hojas son las que descargó el calificador en pantalla, sin editar. No contienen imágenes ni etiquetas.

Para repetir el cálculo, desde la raíz del repositorio:

```
python codigo/review7/kappa_inspection_v2.py resultados/review7/inspeccion_ciega_v2/respuestas_evaluador_A_2026-11-17_18min.csv resultados/review7/inspeccion_ciega_v2/respuestas_evaluador_B_2026-11-17_28min.csv resultados/review7/inspeccion_ciega_v2/respuestas_evaluador_B_2026-11-17_44min.csv resultados/review7/inspeccion_ciega_v2/clave_pares_v2.csv
```

`clave_pares_v2.csv` es la clave del kit (tipo de cada par, identificadores de las dos imágenes y medidas del detector). Los evaluadores no la vieron. `inspeccion_ciega_v2.json` es el resumen que escribe el script. Las imágenes de los pares no están en el repositorio.

## Hojas recibidas

Se recibieron tres hojas completas (356 respuestas cada una), no dos. Dos de ellas se guardaron con la letra B. En este documento se nombran E1, E2 y E3 por orden de inicio.

| Evaluador | Archivo | Inicio y fin (hora de Lima) | Minutos de trabajo | Segundos por par | SHA-256 |
|---|---|---|---|---|---|
| E1 | `respuestas_evaluador_A_2026-11-17_18min.csv` | 15:40 a 15:59 | 18 | 3.0 | `29d3c6fbfe6e870bbe239acba2d442a1838708cb2b03f20f9ea7533b83bae1c5` |
| E2 | `respuestas_evaluador_B_2026-11-17_28min.csv` | 16:26 a 17:12 | 28 | 4.7 | `e913e7124adb2145f3c3284480f82d12b6eda02a2618261842089ef644c49016` |
| E3 | `respuestas_evaluador_B_2026-11-17_44min.csv` | 17:01 a 17:46 | 44 | 7.4 | `1a22e7d44bbae2ecd1479c6a727e7d21b48eb919c362dc9f014c9b88525662fe` |

**Falta que el equipo confirme** quién llenó cada hoja, que E2 y E3 son dos personas distintas, y si alguno había visto antes las hojas de contacto de los pares aceptados o corrió la cuarta etapa.

## Pares juzgados «mismo ojo»

Cada celda da: mismo ojo / ojos distintos / no puedo decidir, y entre paréntesis el porcentaje de «mismo ojo» entre los pares que el evaluador decidió.

| Tipo de par | n | E1 | E2 | E3 |
|---|---|---|---|---|
| Aceptados por la cuarta etapa | 168 | 167 / 0 / 1 (100 %) | 165 / 3 / 0 (98.2 %) | 163 / 4 / 1 (97.6 %) |
| Rechazados parecidos | 168 | 70 / 75 / 23 (48.3 %) | 51 / 115 / 2 (30.7 %) | 7 / 151 / 10 (4.4 %) |
| Al azar (control) | 20 | 3 / 12 / 5 (20.0 %) | 7 / 13 / 0 (35.0 %) | 1 / 18 / 1 (5.3 %) |

Número de evaluadores que juzgaron «mismo ojo» cada par:

| Tipo de par | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| Aceptados | 0 | 0 | 9 | 159 |
| Rechazados parecidos | 77 | 59 | 27 | 5 |
| Al azar | 11 | 7 | 2 | 0 |

Ningún par aceptado fue juzgado «ojos distintos» por dos o más evaluadores (0 de 168; límite superior del intervalo exacto del 95 %: 2.2 %). Siete pares aceptados recibieron un voto de «ojos distintos» y dos un «no puedo decidir».

## Acuerdo

| Comparación | Kappa | IC 95 % (bootstrap) | Acuerdo bruto | Pares decididos por ambos |
|---|---|---|---|---|
| E1 y E2 | 0.576 | 0.480 a 0.668 | 81.8 % | 325 |
| E1 y E3 | 0.581 | 0.499 a 0.669 | 79.7 % | 316 |
| E2 y E3 | 0.702 | 0.630 a 0.770 | 85.1 % | 342 |
| Fleiss, los tres | 0.623 | 0.555 a 0.687 | | 314 |

Acuerdo de cada evaluador con el detector (aceptado = mismo ojo): E1 kappa 0.549 (0.465 a 0.630), E2 0.660 (0.584 a 0.732), E3 0.930 (0.889 a 0.965).

## Cómo leerlo

- **Los enlaces aceptados se sostienen sin usar la etiqueta.** Los tres evaluadores, sin ver la etiqueta ni la decisión del detector, juzgaron «mismo ojo» entre el 97.6 % y el 100 % de los pares aceptados, y ninguno de los 168 fue juzgado «ojos distintos» por la mayoría. Esto es lo que pedía la observación 2.
- **El kappa entre evaluadores es de 0.58 a 0.70 (Fleiss 0.62), por debajo de 0.81.** En la escala de Landis y Koch es un acuerdo de moderado a considerable, no casi perfecto. El desacuerdo no está en los pares aceptados sino en los rechazados y en los pares al azar.
- **La calificación no sirve para medir cuántas copias se le escapan al detector.** Los pares al azar son de imágenes de grupos distintos y deberían juzgarse «ojos distintos». E1 y E2 juzgaron «mismo ojo» el 20 % y el 35 % de ellos; E3, el 5 %. En los rechazados parecidos, la proporción de «mismo ojo» de E2 y E3 no se distingue de la de sus pares al azar (prueba exacta de Fisher, p = 0.80 y p = 1.00), y la de E1 queda en el límite (p = 0.054). Los «mismo ojo» sobre pares rechazados pueden ser, por tanto, error del evaluador y no copias reales.
- **Los evaluadores trabajaron rápido.** Tardaron 18, 28 y 44 minutos, frente a la hora o dos previstas, y el más cuidadoso fue el más lento. E1 marcó 27 de sus 29 «no puedo decidir» en el último cuarto de los pares, lo que sugiere cansancio.

## Desviaciones del protocolo que hay que declarar

1. Calificaron tres personas y no dos. Se reportan las tres hojas, sin elegir entre ellas.
2. Dos hojas se guardaron con la misma letra (B).
3. Los evaluadores son miembros del equipo, no externos.
4. El tiempo de trabajo fue menor que el previsto.

No se cambió ninguna respuesta después de calcular el resultado, y no debe repetirse la calificación para mejorar el kappa.
