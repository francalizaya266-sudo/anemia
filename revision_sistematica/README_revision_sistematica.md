# Revisión sistemática: IA sobre imágenes de conjuntiva para anemia

Protocolo PRISMA 2020 (detallado en el informe del proyecto, sección "Revisión de literatura").

## Pasos
1. Registrar el protocolo en PROSPERO (categoría diagnóstica) u OSF antes de buscar.
2. Buscar en PubMed/MEDLINE, Scopus, Web of Science, IEEE Xplore y Embase, 2015-2026, sin filtro de idioma. Guardar la cadena exacta y la fecha por base (cuaderno de búsqueda).
3. Cadena base (adaptar sintaxis por base):
   (anemia OR anaemia OR hemoglobin OR haemoglobin) AND (conjunctiva* OR palpebral OR eyelid OR pallor) AND (smartphone OR "mobile phone" OR camera OR photograph* OR image*) AND ("machine learning" OR "deep learning" OR "neural network" OR "artificial intelligence" OR classifier OR "computer vision")
4. Exportar a RIS, deduplicar en Zotero, cribar título/resumen por dos revisores en Rayyan (ciego), luego texto completo. Registrar motivos de exclusión. Calcular kappa.
5. Extraer con `plantilla_extraccion_PRISMA.csv` (una fila por estudio; una fila por modelo si el estudio reporta varios y se quiere metaanalizar). La primera fila es un ejemplo ya completado.
6. Riesgo de sesgo: QUADAS-2 (4 dominios) y PROBAST+AI. Columnas incluidas.
7. Síntesis: si >= 5 estudios reportan matriz de confusión o sens/esp con n, metaanálisis bivariado (R: mada o meta4diag) con curva SROC; si no, síntesis narrativa + tabla de fallas metodológicas (fuga por duplicados o aumento, sin ID de paciente, sin IC, sin validación externa, sin XAI cuantificada, sin métricas de despliegue).
8. Diagrama de flujo PRISMA 2020 (plantilla en prisma-statement.org) y lista de verificación PRISMA-DTA como suplemento.

## Columnas clave para nuestra tesis
- `auditoria_duplicados`, `id_paciente_usado`, `aumento_antes_de_dividir`, `tipo_validacion`: cuantifican cuánta literatura tiene el problema que encontramos en CP-AnemiC.
- `XAI_cuantificada`, `tamano_modelo_MB`, `latencia_ms`: muestran el vacío de explicabilidad medida y de factibilidad en borde.
- `ajuste_altitud`: esperamos que sea "no" en todos; justifica el estudio de Puno.
