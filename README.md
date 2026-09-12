# Trading Analitic — Módulo 1

Primera etapa del proyecto Trading Analitic.

## Qué hace

- Conecta con IQ Option.
- Permite seleccionar **un solo activo** mediante radio buttons.
- Permite seleccionar marco de 1, 5 o 15 minutos.
- Permite cargar 45, 60, 120, 180 o 10,000 velas.
- Los datos se mantienen en memoria como un `pandas.DataFrame`.
- No crea un CSV intermedio.
- El botón **Siguiente** deja preparados los datos para el módulo JAPY2.

## Archivos

- `app.py`: interfaz Streamlit.
- `iqoption_service.py`: conexión y descarga paginada.
- `requirements.txt`: dependencias.
- `.streamlit/secrets.toml.example`: ejemplo de configuración segura.

## Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## GitHub + Streamlit Community Cloud

1. Crea un repositorio en GitHub.
2. Sube estos archivos.
3. En Streamlit Community Cloud, crea una app apuntando a `app.py`.
4. Si quieres prellenar el correo sin publicarlo en GitHub, agrega en Secrets:

```toml
IQ_EMAIL = "tu_correo@ejemplo.com"
```

No guardes el password en el repositorio.

## Nota

`iqoptionapi` es una librería no oficial. IQ Option puede cambiar su protocolo
o bloquear funciones, por lo que una actualización futura podría requerir
ajustes en este módulo.
