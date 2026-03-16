"""SECOP Colombia - Chat IA para Contratacion Publica.

Chat con IA (Gemini) que consulta datos de contratacion publica
de Colombia (SECOP I y SECOP II) en lenguaje natural.
"""

import asyncio
import json
import time

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from google.genai.errors import ClientError

from secop_api import (
    DATASETS,
    add_date_filter,
    build_where,
    extract_url,
    query_dataset,
)

st.set_page_config(
    page_title="SecopIA",
    page_icon="🇨🇴",
    layout="wide",
)

st.title("🇨🇴 SecopIA")
st.caption("Realiza consultas directas al SECOP I y SECOP II, potenciadas por Inteligencia Artificial")
st.markdown(
    "> *\"La contratación pública es de todos. Consulta, vigila y exige transparencia.\"*"
)


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

@st.cache_resource
def get_gemini_client():
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        st.error("Falta GEMINI_API_KEY en secrets.")
        st.stop()
    return genai.Client(api_key=api_key)


# ---------------------------------------------------------------------------
# Tools definition for Gemini function calling
# ---------------------------------------------------------------------------

secop_tools = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="buscar_contratos_secop2",
            description=(
                "Busca contratos electronicos en SECOP II (plataforma vigente). "
                "Incluye valores pagados, facturados y pendientes. "
                "Buscar aqui PRIMERO antes de SECOP I."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "entidad": types.Schema(type="STRING", description="Nombre de la entidad contratante"),
                    "proveedor": types.Schema(type="STRING", description="Nombre del proveedor adjudicado"),
                    "nit_proveedor": types.Schema(type="STRING", description="NIT o documento del proveedor"),
                    "objeto": types.Schema(type="STRING", description="Palabras clave del objeto del contrato"),
                    "departamento": types.Schema(type="STRING", description="Departamento"),
                    "modalidad": types.Schema(type="STRING", description="Modalidad de contratacion"),
                    "estado": types.Schema(type="STRING", description="Estado del contrato"),
                    "valor_minimo": types.Schema(type="NUMBER", description="Valor minimo del contrato"),
                    "fecha_desde": types.Schema(type="STRING", description="Fecha inicio rango YYYY-MM-DD"),
                    "fecha_hasta": types.Schema(type="STRING", description="Fecha fin rango YYYY-MM-DD"),
                    "busqueda_texto": types.Schema(type="STRING", description="Busqueda full-text adicional"),
                    "limite": types.Schema(type="INTEGER", description="Maximo de resultados (1-200)"),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="buscar_procesos_secop2",
            description=(
                "Busca procesos de contratacion en SECOP II (plataforma vigente). "
                "Incluye info de entidades, proveedores y adjudicaciones."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "entidad": types.Schema(type="STRING", description="Nombre de la entidad"),
                    "proveedor": types.Schema(type="STRING", description="Nombre del proveedor adjudicado"),
                    "nit_proveedor": types.Schema(type="STRING", description="NIT del proveedor adjudicado"),
                    "objeto": types.Schema(type="STRING", description="Palabras clave del objeto"),
                    "departamento": types.Schema(type="STRING", description="Departamento"),
                    "modalidad": types.Schema(type="STRING", description="Modalidad de contratacion"),
                    "fase": types.Schema(type="STRING", description="Fase del proceso"),
                    "estado": types.Schema(type="STRING", description="Estado del procedimiento"),
                    "fecha_desde": types.Schema(type="STRING", description="Fecha inicio YYYY-MM-DD"),
                    "fecha_hasta": types.Schema(type="STRING", description="Fecha fin YYYY-MM-DD"),
                    "busqueda_texto": types.Schema(type="STRING", description="Busqueda full-text"),
                    "limite": types.Schema(type="INTEGER", description="Maximo de resultados (1-200)"),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="buscar_por_persona",
            description=(
                "Busca en TODOS los datasets SECOP por NIT/cedula o nombre. "
                "USAR PRIMERO si se tiene el documento de la persona o empresa. "
                "Es la herramienta mas completa para investigar un contratista."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "documento": types.Schema(type="STRING", description="NIT o cedula"),
                    "nombre": types.Schema(type="STRING", description="Nombre o razon social"),
                    "limite": types.Schema(type="INTEGER", description="Maximo resultados por dataset (1-100)"),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="buscar_proveedores",
            description="Busca proveedores registrados en SECOP II por nombre, NIT o ubicacion.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "nombre": types.Schema(type="STRING", description="Nombre del proveedor"),
                    "nit": types.Schema(type="STRING", description="NIT del proveedor"),
                    "departamento": types.Schema(type="STRING", description="Departamento"),
                    "ciudad": types.Schema(type="STRING", description="Ciudad"),
                    "limite": types.Schema(type="INTEGER", description="Maximo de resultados (1-200)"),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="agregaciones_contratacion",
            description=(
                "Agrega contratos SECOP II por proveedor, entidad, departamento o modalidad. "
                "Retorna totales: numero de contratos, valor total, valor pagado."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "agrupar_por": types.Schema(
                        type="STRING",
                        description="Campo para agrupar: 'proveedor', 'entidad', 'departamento', 'modalidad'",
                    ),
                    "entidad": types.Schema(type="STRING", description="Filtrar por entidad"),
                    "proveedor": types.Schema(type="STRING", description="Filtrar por proveedor"),
                    "departamento": types.Schema(type="STRING", description="Filtrar por departamento"),
                    "fecha_desde": types.Schema(type="STRING", description="Fecha inicio YYYY-MM-DD"),
                    "fecha_hasta": types.Schema(type="STRING", description="Fecha fin YYYY-MM-DD"),
                    "limite": types.Schema(type="INTEGER", description="Maximo de grupos (1-50)"),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="buscar_secop1",
            description=(
                "Busca en SECOP I (historico, antes de ~2020). "
                "Solo usar si SECOP II no tiene resultados o el contrato es antiguo."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "entidad": types.Schema(type="STRING", description="Nombre de la entidad"),
                    "contratista": types.Schema(type="STRING", description="Nombre del contratista"),
                    "identificacion_contratista": types.Schema(type="STRING", description="Cedula o NIT"),
                    "objeto": types.Schema(type="STRING", description="Palabras clave del objeto"),
                    "departamento": types.Schema(type="STRING", description="Departamento"),
                    "modalidad": types.Schema(type="STRING", description="Modalidad"),
                    "fecha_desde": types.Schema(type="STRING", description="Fecha inicio YYYY-MM-DD"),
                    "fecha_hasta": types.Schema(type="STRING", description="Fecha fin YYYY-MM-DD"),
                    "busqueda_texto": types.Schema(type="STRING", description="Busqueda full-text"),
                    "limite": types.Schema(type="INTEGER", description="Maximo de resultados (1-200)"),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="contar_personas_contratadas",
            description=(
                "Cuenta personas contratadas y valores por anio para una entidad. "
                "Ideal para KPIs de contratacion. Retorna personas unicas, total contratos, "
                "valor total y valor pagado por cada anio."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "entidad": types.Schema(type="STRING", description="Nombre de la entidad contratante"),
                    "anios": types.Schema(
                        type="ARRAY",
                        items=types.Schema(type="INTEGER"),
                        description="Lista de anios a consultar, ej: [2024, 2025, 2026]",
                    ),
                    "solo_personas_naturales": types.Schema(
                        type="BOOLEAN",
                        description="True para solo personas naturales (cedulas), False para incluir empresas (NIT)",
                    ),
                },
                required=["entidad", "anios"],
            ),
        ),
        types.FunctionDeclaration(
            name="resumen_contratacion",
            description=(
                "Resumen condensado de contratos con campos clave: entidad, proveedor, "
                "objeto, valor, estado, fecha. Ideal para explorar antes de pedir detalle completo."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "entidad": types.Schema(type="STRING", description="Nombre de la entidad contratante"),
                    "proveedor": types.Schema(type="STRING", description="Nombre del proveedor"),
                    "nit_proveedor": types.Schema(type="STRING", description="NIT del proveedor"),
                    "departamento": types.Schema(type="STRING", description="Departamento"),
                    "objeto": types.Schema(type="STRING", description="Palabras clave del objeto"),
                    "fecha_desde": types.Schema(type="STRING", description="Fecha inicio YYYY-MM-DD"),
                    "fecha_hasta": types.Schema(type="STRING", description="Fecha fin YYYY-MM-DD"),
                    "limite": types.Schema(type="INTEGER", description="Maximo de resultados (1-200)"),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="consulta_libre",
            description=(
                "Consulta libre con SoQL sobre cualquier dataset SECOP. "
                "Para consultas avanzadas. Permite escribir clausulas SoQL directamente."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "dataset": types.Schema(
                        type="STRING",
                        description="Dataset: 'secop1_procesos', 'secop2_procesos', 'secop2_contratos', 'secop2_proveedores'",
                    ),
                    "where": types.Schema(type="STRING", description="Clausula SoQL $where"),
                    "select": types.Schema(type="STRING", description="Campos a retornar separados por coma"),
                    "order": types.Schema(type="STRING", description="Ordenamiento"),
                    "busqueda_texto": types.Schema(type="STRING", description="Busqueda full-text"),
                    "limite": types.Schema(type="INTEGER", description="Maximo de resultados (1-1000)"),
                },
                required=["dataset"],
            ),
        ),
    ])
]


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def gemini_generate(client, **kwargs):
    """Llama a Gemini con retry automatico ante rate limit (429)."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(**kwargs)
        except ClientError as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                st.warning(f"Rate limit alcanzado. Reintentando en {wait}s...")
                time.sleep(wait)
            else:
                raise


def execute_tool(name: str, args: dict) -> list[dict]:
    """Ejecuta una herramienta SECOP y retorna los resultados crudos."""

    if name == "buscar_contratos_secop2":
        filters = {
            "nombre_entidad": args.get("entidad", ""),
            "proveedor_adjudicado": args.get("proveedor", ""),
            "documento_proveedor": args.get("nit_proveedor") or None,
            "objeto_del_contrato": args.get("objeto", ""),
            "departamento": args.get("departamento", ""),
            "modalidad_de_contratacion": args.get("modalidad", ""),
            "valor_del_contrato": args.get("valor_minimo"),
        }
        where = build_where(filters)
        fd, fh = args.get("fecha_desde", ""), args.get("fecha_hasta", "")
        if fd or fh:
            where = add_date_filter(where, "fecha_de_firma", fd, fh)
        return run_async(query_dataset(
            "secop2_contratos", where=where,
            q=args.get("busqueda_texto") or None,
            limit=min(args.get("limite", 50), 200),
            order="valor_del_contrato DESC",
        ))

    elif name == "buscar_procesos_secop2":
        filters = {
            "entidad": args.get("entidad", ""),
            "nombre_del_proveedor": args.get("proveedor", ""),
            "nit_del_proveedor_adjudicado": args.get("nit_proveedor") or None,
            "descripci_n_del_procedimiento": args.get("objeto", ""),
            "departamento": args.get("departamento", ""),
            "modalidad_de_contratacion": args.get("modalidad", ""),
            "fase": args.get("fase", ""),
            "estado_del_procedimiento": args.get("estado", ""),
        }
        where = build_where(filters)
        fd, fh = args.get("fecha_desde", ""), args.get("fecha_hasta", "")
        if fd or fh:
            where = add_date_filter(where, "fecha_de_publicacion_del", fd, fh)
        return run_async(query_dataset(
            "secop2_procesos", where=where,
            q=args.get("busqueda_texto") or None,
            limit=min(args.get("limite", 50), 200),
            order="valor_total_adjudicacion DESC",
        ))

    elif name == "buscar_por_persona":
        documento = args.get("documento", "")
        nombre = args.get("nombre", "")
        cap = min(args.get("limite", 20), 100)
        searches = {
            "secop2_procesos": {
                "nit_del_proveedor_adjudicado": documento or None,
                "nombre_del_proveedor": nombre or None,
            },
            "secop2_contratos": {
                "documento_proveedor": documento or None,
                "proveedor_adjudicado": nombre or None,
            },
            "secop2_proveedores": {
                "nit_proveedor": documento or None,
                "nombre_proveedor": nombre or None,
            },
            "secop1_procesos": {
                "identificacion_del_contratista": documento or None,
                "nom_razon_social_contratista": nombre or None,
            },
        }
        all_rows = []
        for ds_key, filters in searches.items():
            try:
                where = build_where(filters)
                rows = run_async(query_dataset(ds_key, where=where, limit=cap))
                for r in rows:
                    r["_dataset"] = DATASETS[ds_key]["nombre"]
                all_rows.extend(rows)
            except Exception:
                pass
        return all_rows

    elif name == "buscar_proveedores":
        filters = {
            "nombre_proveedor": args.get("nombre", ""),
            "nit_proveedor": args.get("nit") or None,
            "departamento": args.get("departamento", ""),
            "ciudad": args.get("ciudad", ""),
        }
        where = build_where(filters)
        return run_async(query_dataset(
            "secop2_proveedores", where=where,
            limit=min(args.get("limite", 50), 200),
        ))

    elif name == "agregaciones_contratacion":
        group_fields = {
            "proveedor": "proveedor_adjudicado",
            "entidad": "nombre_entidad",
            "departamento": "departamento",
            "modalidad": "modalidad_de_contratacion",
        }
        agrupar = args.get("agrupar_por", "entidad")
        group_col = group_fields.get(agrupar, "nombre_entidad")
        filters = {
            "nombre_entidad": args.get("entidad", ""),
            "proveedor_adjudicado": args.get("proveedor", ""),
            "departamento": args.get("departamento", ""),
        }
        where = build_where(filters)
        fd, fh = args.get("fecha_desde", ""), args.get("fecha_hasta", "")
        if fd or fh:
            where = add_date_filter(where, "fecha_de_firma", fd, fh)
        select = (
            f"{group_col}, count(*) as total_contratos, "
            f"sum(valor_del_contrato) as valor_total, "
            f"sum(valor_pagado) as valor_total_pagado"
        )
        return run_async(query_dataset(
            "secop2_contratos", where=where, select=select,
            group=group_col, order="valor_total DESC",
            limit=min(args.get("limite", 20), 50),
        ))

    elif name == "buscar_secop1":
        filters = {
            "nombre_entidad": args.get("entidad", ""),
            "nom_razon_social_contratista": args.get("contratista", ""),
            "identificacion_del_contratista": args.get("identificacion_contratista") or None,
            "detalle_del_objeto_a_contratar": args.get("objeto", ""),
            "departamento_entidad": args.get("departamento", ""),
            "modalidad_de_contratacion": args.get("modalidad", ""),
        }
        where = build_where(filters)
        fd, fh = args.get("fecha_desde", ""), args.get("fecha_hasta", "")
        if fd or fh:
            where = add_date_filter(where, "fecha_de_firma_del_contrato", fd, fh)
        return run_async(query_dataset(
            "secop1_procesos", where=where,
            q=args.get("busqueda_texto") or None,
            limit=min(args.get("limite", 50), 200),
            order="cuantia_contrato DESC",
        ))

    elif name == "contar_personas_contratadas":
        entidad = args.get("entidad", "")
        anios = args.get("anios", [])
        solo_nat = args.get("solo_personas_naturales", True)
        tipos_persona = (
            "tipodocproveedor = 'Cédula de Ciudadanía' "
            "OR tipodocproveedor = 'Cédula de Extranjería'"
        )

        async def _query_years():
            results = []
            for year in sorted(anios):
                year = int(year)
                where_parts = [
                    f"upper(nombre_entidad) like upper('%{entidad.replace(chr(39), chr(39)*2)}%')",
                    f"fecha_de_firma >= '{year}-01-01T00:00:00.000'",
                    f"fecha_de_firma <= '{year}-12-31T23:59:59.999'",
                ]
                if solo_nat:
                    where_parts.append(f"({tipos_persona})")
                where = " AND ".join(where_parts)
                rows = await query_dataset(
                    "secop2_contratos",
                    where=where,
                    select=(
                        "count(*) as total_contratos, "
                        "count(distinct documento_proveedor) as personas_unicas, "
                        "sum(valor_del_contrato) as valor_total, "
                        "sum(valor_pagado) as valor_pagado"
                    ),
                    limit=1,
                )
                r = rows[0] if rows else {}
                results.append({
                    "anio": year,
                    "personas_unicas": int(r.get("personas_unicas", 0)),
                    "total_contratos": int(r.get("total_contratos", 0)),
                    "valor_total": float(r.get("valor_total", 0)),
                    "valor_pagado": float(r.get("valor_pagado", 0)),
                })
            return results

        return run_async(_query_years())

    elif name == "resumen_contratacion":
        filters = {
            "nombre_entidad": args.get("entidad", ""),
            "proveedor_adjudicado": args.get("proveedor", ""),
            "documento_proveedor": args.get("nit_proveedor") or None,
            "objeto_del_contrato": args.get("objeto", ""),
            "departamento": args.get("departamento", ""),
        }
        where = build_where(filters)
        fd, fh = args.get("fecha_desde", ""), args.get("fecha_hasta", "")
        if fd or fh:
            where = add_date_filter(where, "fecha_de_firma", fd, fh)
        return run_async(query_dataset(
            "secop2_contratos", where=where,
            select=(
                "nombre_entidad, proveedor_adjudicado, documento_proveedor, "
                "objeto_del_contrato, valor_del_contrato, valor_pagado, "
                "estado_contrato, fecha_de_firma, departamento, "
                "modalidad_de_contratacion, urlproceso"
            ),
            limit=min(args.get("limite", 50), 200),
            order="valor_del_contrato DESC",
        ))

    elif name == "consulta_libre":
        dataset = args.get("dataset", "")
        if dataset not in DATASETS:
            return [{"error": f"Dataset no valido. Opciones: {', '.join(DATASETS.keys())}"}]
        return run_async(query_dataset(
            dataset,
            where=args.get("where") or None,
            select=args.get("select") or None,
            order=args.get("order") or None,
            q=args.get("busqueda_texto") or None,
            limit=min(args.get("limite", 50), 1000),
        ))

    return []


def rows_to_text(rows: list[dict], max_rows: int = 15) -> str:
    """Convierte resultados a texto para enviar a Gemini."""
    if not rows:
        return "No se encontraron resultados."
    display = rows[:max_rows]
    lines = [f"{len(rows)} resultados encontrados. Mostrando {len(display)}:\n"]
    for i, row in enumerate(display, 1):
        lines.append(f"--- Resultado {i} ---")
        for key, value in row.items():
            if value is None or not str(value).strip():
                continue
            if key in ("urlproceso", "ruta_proceso_en_secop_i"):
                url = extract_url(value)
                if url:
                    lines.append(f"  {key}: {url}")
            else:
                val_str = str(value)
                if len(val_str) > 200:
                    val_str = val_str[:197] + "..."
                lines.append(f"  {key}: {val_str}")
        lines.append("")
    if len(rows) > max_rows:
        lines.append(f"... y {len(rows) - max_rows} resultados mas.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chat UI
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = (
    "Eres un asistente experto en contratacion publica de Colombia. "
    "Ayudas a los usuarios a consultar datos del SECOP (Sistema Electronico de Contratacion Publica). "
    "Los datos provienen de datos.gov.co.\n\n"
    "ESTRATEGIA DE BUSQUEDA:\n"
    "1. Si tienes NIT o cedula: usa buscar_por_persona PRIMERO.\n"
    "2. Si conoces la entidad y objeto: usa buscar_contratos_secop2 o buscar_procesos_secop2.\n"
    "3. SECOP II es la plataforma vigente. Busca SIEMPRE primero en SECOP II.\n"
    "4. Solo busca en SECOP I si no hay resultados en SECOP II o el contrato es anterior a 2020.\n"
    "5. Si una busqueda no da resultados, intenta con busqueda_texto.\n"
    "6. Los nombres de entidades pueden diferir del nombre coloquial.\n\n"
    "Responde siempre en espanol. Incluye las URLs de los procesos cuando esten disponibles. "
    "Si los resultados incluyen valores monetarios, presentalos formateados. "
    "SIEMPRE indica el numero total de filas/registros encontrados al inicio de tu respuesta. "
    "Ejemplo: 'Se encontraron 15 registros en total.'"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "dataframe" in msg:
            hist_col_config = {}
            currency_cols = [
                "valor_del_contrato", "valor_pagado", "valor_pendiente_de_pago",
                "valor_facturado", "cuantia_contrato", "precio_base",
                "valor_total_adjudicacion", "valor_total", "valor_total_pagado",
            ]
            for col in currency_cols:
                if col in msg["dataframe"].columns:
                    hist_col_config[col] = st.column_config.NumberColumn(format="$ %,.0f")
            st.dataframe(msg["dataframe"], use_container_width=True, hide_index=True, column_config=hist_col_config)

# Input del usuario
if prompt := st.chat_input("Pregunta sobre contratacion publica..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            client = get_gemini_client()

            # Construir historial para Gemini
            gemini_history = []
            for msg in st.session_state.messages[:-1]:
                role = "user" if msg["role"] == "user" else "model"
                gemini_history.append(types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg["content"])],
                ))

            response = gemini_generate(
                client,
                model="gemini-2.5-flash",
                contents=gemini_history + [
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=prompt)],
                    )
                ],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    tools=secop_tools,
                    temperature=0.3,
                ),
            )

            # Procesar function calls en loop
            all_rows = []
            max_iterations = 5
            iteration = 0

            while response.candidates and iteration < max_iterations:
                candidate = response.candidates[0]
                parts = candidate.content.parts

                # Buscar function calls
                function_calls = [p for p in parts if p.function_call]
                if not function_calls:
                    break

                # Ejecutar cada function call
                function_responses = []
                for fc_part in function_calls:
                    fc = fc_part.function_call
                    tool_name = fc.name
                    tool_args = dict(fc.args) if fc.args else {}

                    st.caption(f"Consultando: {tool_name}({json.dumps(tool_args, ensure_ascii=False)})")

                    try:
                        rows = execute_tool(tool_name, tool_args)
                        all_rows.extend(rows)
                        result_text = rows_to_text(rows)
                    except Exception as e:
                        result_text = f"Error: {e}"

                    function_responses.append(
                        types.Part.from_function_response(
                            name=tool_name,
                            response={"result": result_text},
                        )
                    )

                # Enviar resultados de vuelta a Gemini
                response = gemini_generate(
                    client,
                    model="gemini-2.5-flash",
                    contents=gemini_history + [
                        types.Content(role="user", parts=[types.Part.from_text(text=prompt)]),
                        candidate.content,
                        types.Content(role="user", parts=function_responses),
                    ],
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        tools=secop_tools,
                        temperature=0.3,
                    ),
                )
                iteration += 1

            # Extraer respuesta final
            final_text = ""
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if part.text:
                        final_text += part.text

            if not final_text:
                final_text = "No pude generar una respuesta. Intenta reformular tu pregunta."

            st.markdown(final_text)

            # Mostrar tabla si hay datos
            msg_data = {"role": "assistant", "content": final_text}
            if all_rows:
                for row in all_rows:
                    for field in ["urlproceso", "ruta_proceso_en_secop_i"]:
                        if field in row:
                            row[field] = extract_url(row[field])
                df = pd.DataFrame(all_rows)
                currency_cols = [
                    "valor_del_contrato", "valor_pagado", "valor_pendiente_de_pago",
                    "valor_facturado", "cuantia_contrato", "precio_base",
                    "valor_total_adjudicacion", "valor_total", "valor_total_pagado",
                ]
                for col in currency_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                # Formatear columnas monetarias como pesos colombianos
                col_config = {}
                for col in currency_cols:
                    if col in df.columns:
                        col_config[col] = st.column_config.NumberColumn(
                            format="$ %,.0f",
                        )
                st.dataframe(df, use_container_width=True, hide_index=True, column_config=col_config)
                msg_data["dataframe"] = df

                csv = df.to_csv(index=False)
                st.download_button(
                    "Descargar CSV",
                    csv,
                    file_name="secop_resultados.csv",
                    mime="text/csv",
                )

            st.session_state.messages.append(msg_data)

# --- Sidebar ---
with st.sidebar:
    st.header("Ejemplos de preguntas")
    st.markdown("""
    - Busca los contratos de la empresa con NIT 900123456
    - Que contratos tiene la Alcaldia de Bogota en 2024?
    - Muestra los proveedores mas grandes de Antioquia
    - Cuanto ha contratado el Ministerio de Educacion?
    - Busca contratos de consultoria en Cundinamarca mayores a 500 millones
    """)
    st.divider()
    if st.button("Limpiar chat"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.markdown("##### Acerca de SecopIA")
    st.markdown("""
    **Fuente de datos**
    - [datos.gov.co](https://www.datos.gov.co/) — Portal de datos abiertos del gobierno colombiano
    - Consultas en tiempo real a la API SODA de Socrata

    **Datasets disponibles**
    - SECOP I — Procesos históricos (antes de ~2020)
    - SECOP II — Procesos de contratación vigentes
    - SECOP II — Contratos electrónicos firmados
    - SECOP II — Proveedores registrados

    **Inteligencia Artificial**
    - Modelo: Gemini 2.5 Flash (Google)
    - Capacidad de function calling para decidir qué consulta ejecutar

    **Proyecto open source**
    - [secop-mcp-server](https://github.com/juandavidsernav/secop-mcp-server) en PyPI
    - Desarrollado por Juan David Serna Valderrama
    """)

    st.divider()
    st.markdown("##### Apoya esta iniciativa")
    st.markdown("""
    SecopIA es una herramienta gratuita y de código abierto, construida con
    la convicción de que el acceso a la información pública fortalece la
    democracia y la veeduría ciudadana.

    Tu apoyo nos ayuda a mantenerla activa, mejorarla y que siga siendo
    gratuita para todos los colombianos.

    ☕ [Apóyanos en Ko-fi](https://ko-fi.com/juandavidsernavalderrama)
    """)
    st.caption("¡Gracias por creer en la transparencia! 🙌")
