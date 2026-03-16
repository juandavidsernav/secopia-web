# 🇨🇴 SecopIA

Herramienta potenciada por Inteligencia Artificial para consultas directas al SECOP I y SECOP II.

> *"La contratación pública es de todos. Consulta, vigila y exige transparencia."*

## Qué es SecopIA

SecopIA es una aplicación web gratuita que permite a cualquier ciudadano colombiano consultar datos de contratación pública en lenguaje natural. Escribe tu pregunta como si hablaras con una persona y la IA se encarga de buscar en los datasets del SECOP.

## Funcionalidades

- **Chat con IA** — Pregunta en lenguaje natural sobre contratos, procesos y proveedores
- **Consulta en tiempo real** — Los datos provienen directamente de [datos.gov.co](https://www.datos.gov.co/) vía API SODA
- **9 herramientas de búsqueda** — Contratos, procesos, proveedores, agregaciones, conteo de personas contratadas y más
- **Descarga CSV** — Exporta los resultados de cualquier consulta
- **Tablas interactivas** — Visualiza y ordena resultados con valores formateados en pesos colombianos

## Datasets disponibles

| Dataset | Descripción |
|---|---|
| SECOP I - Procesos | Datos históricos de contratación (antes de ~2020) |
| SECOP II - Procesos | Procesos de contratación vigentes |
| SECOP II - Contratos | Contratos electrónicos firmados |
| SECOP II - Proveedores | Proveedores registrados en la plataforma |

## Ejemplos de preguntas

- *"Busca los contratos de la empresa con NIT 900123456"*
- *"¿Qué contratos tiene la Alcaldía de Bogotá en 2024?"*
- *"Muestra los proveedores más grandes de Antioquia"*
- *"¿Cuántas personas ha contratado el Ministerio de Educación entre 2023 y 2025?"*
- *"Busca contratos de consultoría en Cundinamarca mayores a 500 millones"*

## Stack técnico

- **Frontend**: [Streamlit](https://streamlit.io/)
- **IA**: Google Gemini 2.5 Flash (function calling)
- **Datos**: API SODA de Socrata / datos.gov.co
- **Backend de consultas**: [secop-mcp-server](https://github.com/juandavidsernav/secop-mcp-server)

## Ejecutar localmente

```bash
# Clonar el repositorio
git clone https://github.com/juandavidsernav/secopia-web.git
cd secopia-web

# Instalar dependencias
pip install -r requirements.txt

# Configurar API key de Gemini
mkdir -p .streamlit
echo 'GEMINI_API_KEY = "tu-api-key"' > .streamlit/secrets.toml

# Ejecutar
streamlit run app.py
```

## Obtener API key de Gemini (gratis)

1. Ve a [Google AI Studio](https://aistudio.google.com/apikey)
2. Crea una API key
3. Agrégala en `.streamlit/secrets.toml`

## Apoya esta iniciativa

SecopIA es gratuita y de código abierto. Tu apoyo ayuda a mantenerla activa y mejorarla.

☕ [Apóyanos en Ko-fi](https://ko-fi.com/juandavidsernavalderrama)

## Licencia

MIT

## Autor

Juan David Serna Valderrama
